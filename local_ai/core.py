import os
import json
import time
import pickle
import psutil
import asyncio
import socket
import requests
import subprocess
from pathlib import Path
from loguru import logger
from typing import Optional, Dict, Any
import pkg_resources
from local_ai.download import download_model_from_filecoin_async

class LocalAIServiceError(Exception):
    """Base exception for Local AI service errors."""
    pass

class ServiceStartError(LocalAIServiceError):
    """Exception raised when service fails to start."""
    pass

class ServiceHealthError(LocalAIServiceError):
    """Exception raised when service health check fails."""
    pass

class ModelNotFoundError(LocalAIServiceError):
    """Exception raised when model file is not found."""
    pass

class LocalAIManager:
    """Manages a local AI service."""
    
    def __init__(self):
        """Initialize the LocalAIManager."""       
        self.pickle_file = Path(os.getenv("RUNNING_SERVICE_FILE", "running_service.pkl"))
        self.loaded_models: Dict[str, Any] = {}
        self.llama_server_path = os.getenv("LLAMA_SERVER")
        if not self.llama_server_path or not os.path.exists(self.llama_server_path):
            raise LocalAIServiceError("llama-server executable not found in LLAMA_SERVER or PATH")
        
    def _get_free_port(self) -> int:
        """Get a free port number."""
        with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as s:
            s.bind(("", 0))
            return s.getsockname()[1]

    def _wait_for_service(self, port: int, timeout: int = 300) -> bool:
        """
        Wait for the AI service to become healthy.

        Args:
            port (int): Port number of the service.
            timeout (int): Maximum time to wait in seconds (default: 300).

        Returns:
            bool: True if service is healthy, False otherwise.

        Raises:
            ServiceHealthError: If service fails to become healthy within timeout.
        """
        health_check_url = f"http://localhost:{port}/health"
        start_time = time.time()
        wait_time = 1  # Initial wait time in seconds
        last_error = None
        
        while time.time() - start_time < timeout:
            try:
                status = requests.get(health_check_url, timeout=5)
                if status.status_code == 200 and (status.json().get("status") == "ok"):
                    logger.debug(f"Service healthy at {health_check_url}")
                    return True
            except requests.exceptions.RequestException as e:
                last_error = str(e)
                logger.debug(f"Health check failed: {last_error}")
            time.sleep(wait_time)
            wait_time = min(wait_time * 2, 60)  # Exponential backoff, max 60s
            
        raise ServiceHealthError(f"Service failed to become healthy within {timeout} seconds. Last error: {last_error}")
    
    def restart(self):
        """
        Restart the currently running AI service.

        Returns:
            bool: True if the service restarted successfully, False otherwise.
        """
        if not self.pickle_file.exists():
            logger.warning("No running AI service to restart.")
            return False
        
        try:
            # Load service details from the pickle file
            with open(self.pickle_file, "rb") as f:
                service_info = pickle.load(f)
            
            hash = service_info.get("hash")
            port = service_info.get("app_port")
            context_length = service_info.get("context_length")

            logger.info(f"Restarting AI service '{hash}' running on port {port}...")

            # Stop the current service
            self.stop()

            # Start the service with the same parameters
            return self.start(hash, port, context_length=context_length)
        except Exception as e:
            logger.error(f"Error restarting AI service: {str(e)}", exc_info=True)
            return False

    def _get_family_template_and_practice(self, model_family: str):
        """Helper to get template and best practice paths based on folder name."""
        return (
            self._get_model_template_path(model_family),
            self._get_model_best_practice_path(model_family)
        )
    
    def _retry_request_json(self, url, retries=3, delay=5, timeout=10):
        """Utility to retry a GET request for JSON data."""
        for attempt in range(retries):
            try:
                response = requests.get(url, timeout=timeout)
                if response.status_code == 200:
                    return response.json()
            except requests.exceptions.RequestException as e:
                logger.warning(f"Attempt {attempt+1} failed for {url}: {e}")
                time.sleep(delay)
        return None

    def start(self, hash: str, port: int = 11434, host: str = "0.0.0.0", context_length: int = 32768) -> bool:
        """
        Start the local AI service in the background.

        Args:
            hash (str): Filecoin hash of the model to download and run.
            port (int): Port number for the AI service (default: 11434).
            host (str): Host address for the AI service (default: "0.0.0.0").
            context_length (int): Context length for the model (default: 32768).

        Returns:
            bool: True if service started successfully, False otherwise.

        Raises:
            ValueError: If hash is not provided when no model is running.
            ModelNotFoundError: If model file is not found.
            ServiceStartError: If service fails to start.
        """
        if not hash:
            raise ValueError("Filecoin hash is required to start the service")

        try:
            logger.info(f"Starting local AI service for model with hash: {hash}")
            
            local_model_path = asyncio.run(download_model_from_filecoin_async(hash))
            local_projector_path = local_model_path + "-projector"
            model_running = self.get_running_model()
            local_ai_port = self._get_free_port()
            if model_running:
                if model_running == hash:
                    logger.warning(f"Model '{hash}' already running on port {port}")
                    return True
                logger.info(f"Stopping existing model '{model_running}' on port {port}")
                self.stop()

            if not os.path.exists(local_model_path):
                raise ModelNotFoundError(f"Model file not found at: {local_model_path}")

            service_metadata = {
                "hash": hash,
                "port": local_ai_port,  # Local AI server port
                "local_text_path": local_model_path,
                "app_port": port,  # FastAPI port
                "context_length": context_length,
                "last_activity": time.time(),
                "multimodal": os.path.exists(local_projector_path),
                "local_projector_path": local_projector_path if os.path.exists(local_projector_path) else None
            }

            # Get the directory of the model file
            model_dir = os.path.dirname(local_model_path)
            metadata_file = os.path.join(model_dir, f"{hash}.json")

            logger.info(f"metadata_file: {metadata_file}")
            folder_name = ""

            # Check if metadata file exists
            if os.path.exists(metadata_file):
                try:
                    with open(metadata_file, "r") as f:
                        metadata = json.load(f)
                        service_metadata["family"] = metadata.get("family", "")
                        folder_name = metadata.get("folder_name", "")
                        logger.info(f"Loaded metadata from {metadata_file}")
                except Exception as e:
                    logger.error(f"Error loading metadata file: {e}")
                    metadata_file = None
            else:
                filecoin_url = f"https://gateway.lighthouse.storage/ipfs/{hash}"
                response_json = self._retry_request_json(filecoin_url, retries=3, delay=5, timeout=10)
                folder_name = response_json.get("folder_name", "")
                service_metadata["family"] = response_json.get("family", "")
                try:
                    with open(metadata_file, "w") as f:
                        json.dump(response_json, f)
                    logger.info(f"Saved metadata to {metadata_file}")
                except Exception as e:
                    logger.error(f"Error saving metadata file: {e}")

            print(
                "folder_name: ", folder_name
            )

            if "gemma" in folder_name.lower():
                template_path, best_practice_path = self._get_family_template_and_practice("gemma")
                # Gemma models are memory intensive, so we reduce the context length
                context_length = context_length // 2
                running_ai_command = self._build_ai_command(
                    local_model_path, local_ai_port, host, context_length, template_path
                )
            elif "deepseek-r1-0528-qwen3" in folder_name.lower():
                print(
                    "deepseek-r1-0528-qwen3 in folder_name"
                )
                template_path, best_practice_path = self._get_family_template_and_practice("deepseek-r1-0528-qwen3")
                running_ai_command = self._build_ai_command(
                    local_model_path, local_ai_port, host, context_length, template_path, best_practice_path
                )
            elif "qwen25" in folder_name.lower():
                template_path, best_practice_path = self._get_family_template_and_practice("qwen25")
                running_ai_command = self._build_ai_command(
                    local_model_path, local_ai_port, host, context_length, template_path, best_practice_path
                )
            elif "qwen3" in folder_name.lower():
                template_path, best_practice_path = self._get_family_template_and_practice("qwen3")
                running_ai_command = self._build_ai_command(
                    local_model_path, local_ai_port, host, context_length, template_path, best_practice_path
                )
            elif "llama" in folder_name.lower():
                template_path, best_practice_path = self._get_family_template_and_practice("llama")
                running_ai_command = self._build_ai_command(
                    local_model_path, local_ai_port, host, context_length, template_path, best_practice_path
                )
            else:
                running_ai_command = self._build_ai_command(
                    local_model_path, local_ai_port, host, context_length
                )

            if service_metadata["multimodal"]:
                running_ai_command.extend([
                    "--mmproj", str(local_projector_path)
                ])

            logger.info(f"Starting process: {' '.join(running_ai_command)}")
            service_metadata["running_ai_command"] = running_ai_command
            # Create log files for stdout and stderr for AI process
            os.makedirs("logs", exist_ok=True)
            ai_log_stderr = Path(f"logs/ai.log")
            ai_process = None
            try:
                with open(ai_log_stderr, 'w') as stderr_log:
                    ai_process = subprocess.Popen(
                        running_ai_command,
                        stderr=stderr_log,
                        preexec_fn=os.setsid
                    )
                logger.info(f"AI logs written to {ai_log_stderr}")
            except Exception as e:
                logger.error(f"Error starting AI service: {str(e)}", exc_info=True)
                return False
    
            if not self._wait_for_service(local_ai_port):
                logger.error(f"Service failed to start within 600 seconds")
                ai_process.terminate()
                return False

            # start the FastAPI app in the background           
            uvicorn_command = [
                "uvicorn",
                "local_ai.apis:app",
                "--host", host,
                "--port", str(port),
                "--log-level", "info"
            ]
            logger.info(f"Starting process: {' '.join(uvicorn_command)}")
            # Create log files for stdout and stderr
            os.makedirs("logs", exist_ok=True)
            api_log_stderr = Path(f"logs/api.log")
            try:
                with open(api_log_stderr, 'w') as stderr_log:
                    apis_process = subprocess.Popen(
                        uvicorn_command,
                        stderr=stderr_log,
                        preexec_fn=os.setsid
                    )
                logger.info(f"API logs written to {api_log_stderr}")
            except Exception as e:
                logger.error(f"Error starting FastAPI app: {str(e)}", exc_info=True)
                ai_process.terminate()
                return False
            
            if not self._wait_for_service(port):
                logger.error(f"API service failed to start within 600 seconds")
                ai_process.terminate()
                apis_process.terminate()
                return False

            logger.info(f"Service started on port {port} for model: {hash}")

            service_metadata["pid"] = ai_process.pid
            service_metadata["app_pid"] = apis_process.pid
            projector_path = f"{local_model_path}-projector"    
            if os.path.exists(projector_path):
                service_metadata["multimodal"] = True
                service_metadata["local_projector_path"] = projector_path

            self._dump_running_service(service_metadata)    

            # update service metadata to the FastAPI app
            try:
                update_url = f"http://localhost:{port}/update"
                response = requests.post(update_url, json=service_metadata, timeout=10)
                response.raise_for_status()  # Raise exception for HTTP error responses
            except requests.exceptions.RequestException as e:
                logger.error(f"Failed to update service metadata: {str(e)}")
                # Stop the partially started service
                self.stop()
                return False
            
            return True

        except Exception as e:
            logger.error(f"Error starting AI service: {str(e)}", exc_info=True)
            return False
        
    def _dump_running_service(self, metadata: dict):
        """Dump the running service details to a file."""
        try:
            with open(self.pickle_file, "wb") as f:
                pickle.dump(metadata, f)
        except Exception as e:
            logger.error(f"Error dumping running service: {str(e)}", exc_info=True)
            return False

    def get_running_model(self) -> Optional[str]:
        """
        Get currently running model hash if the service is healthy.

        Returns:
            Optional[str]: Running model hash or None if no healthy service exists.
        """
        if not self.pickle_file.exists():
            return None

        try:
            # Load service info from pickle file
            with open(self.pickle_file, "rb") as f:
                service_info = pickle.load(f)   
            model_hash = service_info.get("hash")      
            return model_hash

        except Exception as e:
            logger.error(f"Error getting running model: {str(e)}")
            return None
    
    def stop(self) -> bool:
        """
        Stop the running AI service.

        Returns:
            bool: True if the service stopped successfully, False otherwise.
        """
        if not os.path.exists(self.pickle_file):
            logger.warning("No running AI service to stop.")
            return False

        try:
            # Load service details from the pickle file
            with open(self.pickle_file, "rb") as f:
                service_info = pickle.load(f)
            
            hash = service_info.get("hash")
            pid = service_info.get("pid")
            app_pid = service_info.get("app_pid")
            app_port = service_info.get("app_port")

            logger.info(f"Stopping AI service '{hash}' running on port {app_port} (PID: {app_pid})...")
            
            def terminate_process(pid, process_name, max_retries=3):
                """Helper function to terminate a process with retries."""
                if not psutil.pid_exists(pid):
                    return True
                
                for attempt in range(max_retries):
                    try:
                        process = psutil.Process(pid)
                        process.terminate()
                        process.wait(timeout=30)  # Reduced timeout per attempt
                        
                        if not process.is_running():
                            return True
                            
                        logger.warning(f"Process {process_name} (PID: {pid}) did not terminate on attempt {attempt + 1}, forcing kill.")
                        process.kill()
                        process.wait(timeout=10)
                        
                        if not process.is_running():
                            return True
                            
                        time.sleep(1)  # Wait before next retry
                    except psutil.NoSuchProcess:
                        return True
                    except Exception as e:
                        logger.error(f"Error terminating {process_name} (PID: {pid}) on attempt {attempt + 1}: {str(e)}")
                        time.sleep(1)  # Wait before next retry
                
                logger.error(f"Failed to terminate {process_name} (PID: {pid}) after {max_retries} attempts")
                return False

            # Terminate AI process
            if not terminate_process(pid, "AI service"):
                logger.error("Failed to terminate AI service process")
                return False

            # Terminate FastAPI app
            if not terminate_process(app_pid, "API service"):
                logger.error("Failed to terminate API service process")
                return False
            
            try:
                # remove the pickle file
                os.remove(self.pickle_file)
            except Exception as e:
                logger.error(f"Error removing pickle file: {str(e)}")
                return False

            logger.info("AI service stopped successfully.")
            return True

        except Exception as e:
            logger.error(f"Error stopping AI service: {str(e)}", exc_info=True)
            return False

    def _get_model_template_path(self, model_family: str) -> str:
        """Get the template path for a specific model family."""
        chat_template_path = pkg_resources.resource_filename("local_ai", f"examples/templates/{model_family}.jinja")
        # check if the template file exists
        if not os.path.exists(chat_template_path):
            return None
        return chat_template_path

    def _get_model_best_practice_path(self, model_family: str) -> str:
        """Get the best practices for a specific model family."""
        best_practice_path = pkg_resources.resource_filename("local_ai", f"examples/best_practices/{model_family}.json")
        # check if the best practices file exists
        if not os.path.exists(best_practice_path):
            return None
        return best_practice_path

    def _build_ai_command(self, model_path: str, port: int, host: str, context_length: int, template_path: Optional[str] = None, best_practice_path: Optional[str] = None) -> list:
        """Build the AI command with common parameters."""
        command = [
            self.llama_server_path,
            "--model", str(model_path),
            "--port", str(port),
            "--host", host,
            "-c", str(context_length),
            "-fa",
            "--pooling", "mean",
            "--no-webui",
            "-ngl", "-1",
            "--no-mmap",
            "--mlock",
            "--jinja"
        ]
        
        if template_path:
            command.extend(["--chat-template-file", template_path])
        
        if best_practice_path:
            with open(best_practice_path, "r") as f:
                best_practice = json.load(f)
                for key, value in best_practice.items():
                    command.extend([f"--{key}", str(value)])
        return command