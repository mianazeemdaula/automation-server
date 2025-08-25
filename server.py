import asyncio
import os
import uuid
import tempfile
import shutil
import re
import json
import aiohttp
from typing import List, Dict, Optional, Literal, Tuple
from datetime import datetime
from enum import Enum
from concurrent.futures import ThreadPoolExecutor
import threading

from fastapi import FastAPI, HTTPException, BackgroundTasks
from pydantic import BaseModel
from gradio_client import Client
from pydub import AudioSegment
import uvicorn

# Models
class TaskStatus(str, Enum):
    PENDING = "pending"
    PROCESSING = "processing"
    COMPLETED = "completed"
    FAILED = "failed"

class Scene(BaseModel):
    id: str
    text: str
    emotion: str

class TTSRequest(BaseModel):
    scenes: List[Scene]
    voice: Literal['alloy', 'echo', 'fable', 'onyx', 'nova', 'shimmer', 'coral', 'verse', 'ballad', 'ash', 'sage', 'amuch', 'dan'] = "alloy"
    method: Literal['gradio_client', 'http_api'] = "gradio_client"  # New field to choose method

class TaskResponse(BaseModel):
    task_id: str
    status: TaskStatus
    message: str

class TaskStatusResponse(BaseModel):
    task_id: str
    status: TaskStatus
    progress: str
    download_url: Optional[str] = None
    error_message: Optional[str] = None

# Task storage (in production, use Redis or database)
tasks_storage: Dict[str, Dict] = {}
tasks_lock = threading.Lock()

# FastAPI app
app = FastAPI(title="TTS Audio Generator", version="1.0.0")

# Configuration
MAX_CONCURRENT_GRADIO_REQUESTS = 10  # Increased for HTTP method
GRADIO_API_BASE = "https://nihalgazi-text-to-speech-unlimited.hf.space/gradio_api/call"
OUTPUT_DIR = "generated_audio"
TEMP_DIR = "temp"
os.makedirs(OUTPUT_DIR, exist_ok=True)
os.makedirs(TEMP_DIR, exist_ok=True)

# Thread pool for concurrent Gradio requests
executor = ThreadPoolExecutor(max_workers=MAX_CONCURRENT_GRADIO_REQUESTS)

class TTSProcessor:
    def __init__(self, task_id: str):
        # Create task-specific temp directory
        self.task_temp_dir = os.path.join(TEMP_DIR, task_id)
        os.makedirs(self.task_temp_dir, exist_ok=True)
        
        self.gradio_client = Client("NihalGazi/Text-To-Speech-Unlimited", download_files=self.task_temp_dir)
    
    def clean_text(self, text: str) -> str:
        """Clean text by escaping quotes and handling special characters"""
        # Replace double quotes with escaped quotes
        cleaned_text = text.replace('"', '\\"')
        # Remove or replace other problematic characters if needed
        # cleaned_text = re.sub(r'[\x00-\x1f\x7f-\x9f]', '', cleaned_text)  # Remove control characters
        # # Limit text length to prevent API issues (adjust as needed)
        # if len(cleaned_text) > 500:  # Adjust this limit based on API constraints
        #     cleaned_text = cleaned_text[:497] + "..."
        return cleaned_text

    def combine_audio_files_ordered(self, scene_audio_map: Dict[str, Optional[str]], scenes: List, output_path: str) -> bool:
        """Combine audio files in the order of scenes, skipping None entries"""
        try:
            from pydub import AudioSegment
            
            combined = AudioSegment.empty()
            files_combined = 0
            
            for scene in scenes:
                audio_file_path = scene_audio_map.get(scene.id)
                
                if audio_file_path and os.path.exists(audio_file_path):
                    print(f"Adding scene {scene.id} audio to combination: {audio_file_path}")
                    try:
                        audio = AudioSegment.from_file(audio_file_path)
                        combined += audio
                        # Add small pause between scenes
                        combined += AudioSegment.silent(duration=500)  # 500ms pause
                        files_combined += 1
                    except Exception as e:
                        print(f"Error processing audio file {audio_file_path}: {e}")
                else:
                    print(f"Skipping scene {scene.id} - no valid audio file")
            
            if files_combined > 0:
                combined.export(output_path, format="mp3")
                print(f"Successfully combined {files_combined} audio files to: {output_path}")
                return True
            else:
                print("No valid audio files to combine")
                return False
                
        except Exception as e:
            print(f"Failed to combine audio files: {e}")
            return False
    
    async def generate_single_audio_http(self, session: aiohttp.ClientSession, text: str, voice: str, emotion: str, scene_id: str) -> Optional[str]:
        """Generate audio for a single scene using HTTP API"""
        max_retries = 3
        
        for attempt in range(max_retries):
            try:
                cleaned_text = self.clean_text(text)
                print(f"🌐 HTTP: Generating audio for scene {scene_id} (attempt {attempt + 1}/{max_retries})")
                
                # Step 1: Submit the request and get event_id
                payload = {
                    "data": [
                        cleaned_text,  # prompt
                        voice,         # voice
                        emotion,       # emotion
                        True,          # use_random_seed
                        12345          # specific_seed
                    ]
                }
                
                async with session.post(
                    "https://nihalgazi-text-to-speech-unlimited.hf.space/gradio_api/call/text_to_speech_app",
                    json=payload,
                    timeout=aiohttp.ClientTimeout(total=30)
                ) as response:
                    if response.status == 200:
                        result = await response.json()
                        event_id = result.get("event_id")
                        
                        if not event_id:
                            print(f"✗ HTTP: No event_id received for scene {scene_id}")
                            continue
                        
                        print(f"🌐 HTTP: Got event_id {event_id} for scene {scene_id}")
                        
                        # Step 2: Poll for the result using the event_id
                        result_url = f"https://nihalgazi-text-to-speech-unlimited.hf.space/gradio_api/call/text_to_speech_app/{event_id}"
                        
                        # Wait a bit before polling
                        await asyncio.sleep(2)
                        
                        # Poll for result with timeout
                        max_polls = 30  # 30 seconds max wait
                        for poll in range(max_polls):
                            try:
                                async with session.get(result_url, timeout=aiohttp.ClientTimeout(total=10)) as result_response:
                                    if result_response.status == 200:
                                        result_data = await result_response.json()
                                        
                                        # Parse the Server-Sent Events format
                                        if result_data and len(result_data) > 0:
                                            event_data = result_data[0].get("data", "")
                                            
                                            # Check if it's a complete event
                                            if "event: complete" in event_data:
                                                # Extract the JSON data after "data: "
                                                try:
                                                    json_start = event_data.find("data: [")
                                                    if json_start != -1:
                                                        json_data = event_data[json_start + 6:].strip()
                                                        # Remove trailing newlines
                                                        json_data = json_data.rstrip('\n')
                                                        
                                                        parsed_data = json.loads(json_data)
                                                        
                                                        if parsed_data and len(parsed_data) > 0:
                                                            file_info = parsed_data[0]
                                                            
                                                            if isinstance(file_info, dict) and "url" in file_info:
                                                                file_url = file_info["url"]
                                                                
                                                                # Download the audio file
                                                                async with session.get(file_url, timeout=aiohttp.ClientTimeout(total=30)) as file_response:
                                                                    if file_response.status == 200:
                                                                        file_content = await file_response.read()
                                                                        
                                                                        # Save to temp file
                                                                        temp_filename = f"{scene_id}_{attempt}.mp3"
                                                                        temp_filepath = os.path.join(self.task_temp_dir, temp_filename)
                                                                        
                                                                        with open(temp_filepath, "wb") as f:
                                                                            f.write(file_content)
                                                                        
                                                                        if os.path.exists(temp_filepath) and os.path.getsize(temp_filepath) > 0:
                                                                            print(f"✓ HTTP: Generated audio for scene {scene_id}: {temp_filepath}")
                                                                            return temp_filepath
                                                                        else:
                                                                            print(f"✗ HTTP: Downloaded file is empty for scene {scene_id}")
                                                                    else:
                                                                        print(f"✗ HTTP: Failed to download audio file for scene {scene_id}: HTTP {file_response.status}")
                                                            else:
                                                                print(f"✗ HTTP: No file URL in response for scene {scene_id}: {file_info}")
                                                        else:
                                                            print(f"✗ HTTP: Empty or null data in response for scene {scene_id}")
                                                            break  # Don't retry if data is null (generation failed)
                                                except json.JSONDecodeError as e:
                                                    print(f"✗ HTTP: Failed to parse JSON for scene {scene_id}: {e}")
                                                    print(f"Raw data: {event_data}")
                                                    break
                                            elif "event: error" in event_data:
                                                print(f"✗ HTTP: Error event received for scene {scene_id}: {event_data}")
                                                break
                                            else:
                                                # Still processing, wait a bit more
                                                await asyncio.sleep(1)
                                                continue
                                    else:
                                        print(f"✗ HTTP: Failed to poll result for scene {scene_id}: HTTP {result_response.status}")
                                        break
                            except asyncio.TimeoutError:
                                print(f"⏰ HTTP: Polling timeout for scene {scene_id} (poll {poll + 1})")
                                continue
                            except Exception as e:
                                print(f"✗ HTTP: Polling error for scene {scene_id}: {e}")
                                break
                        
                        print(f"✗ HTTP: Polling completed without success for scene {scene_id}")
                    else:
                        print(f"✗ HTTP: Initial request failed for scene {scene_id}: HTTP {response.status}")
                        
            except asyncio.TimeoutError:
                print(f"✗ HTTP: Timeout for scene {scene_id} (attempt {attempt + 1})")
            except Exception as e:
                print(f"✗ HTTP: Error for scene {scene_id} (attempt {attempt + 1}): {e}")
            
            if attempt < max_retries - 1:
                await asyncio.sleep(2)  # Wait before retry
        
        # Final fallback with null emotion
        if emotion:  # Only try fallback if we had an emotion originally
            try:
                print(f"🌐 HTTP: Final fallback for scene {scene_id} with null emotion")
                cleaned_text = self.clean_text(text)
                
                payload = {
                    "data": [
                        cleaned_text,  # prompt
                        voice,         # voice
                        "",            # null emotion
                        True,          # use_random_seed
                        12345          # specific_seed
                    ]
                }
                
                async with session.post(
                    "https://nihalgazi-text-to-speech-unlimited.hf.space/gradio_api/call/text_to_speech_app",
                    json=payload,
                    timeout=aiohttp.ClientTimeout(total=30)
                ) as response:
                    if response.status == 200:
                        result = await response.json()
                        event_id = result.get("event_id")
                        
                        if event_id:
                            await asyncio.sleep(2)
                            result_url = f"https://nihalgazi-text-to-speech-unlimited.hf.space/gradio_api/call/text_to_speech_app/{event_id}"
                            
                            for poll in range(30):
                                async with session.get(result_url, timeout=aiohttp.ClientTimeout(total=10)) as result_response:
                                    if result_response.status == 200:
                                        result_data = await result_response.json()
                                        
                                        if result_data and len(result_data) > 0:
                                            event_data = result_data[0].get("data", "")
                                            
                                            if "event: complete" in event_data:
                                                json_start = event_data.find("data: [")
                                                if json_start != -1:
                                                    json_data = event_data[json_start + 6:].strip().rstrip('\n')
                                                    
                                                    try:
                                                        parsed_data = json.loads(json_data)
                                                        
                                                        if parsed_data and len(parsed_data) > 0:
                                                            file_info = parsed_data[0]
                                                            
                                                            if isinstance(file_info, dict) and "url" in file_info:
                                                                file_url = file_info["url"]
                                                                
                                                                async with session.get(file_url, timeout=aiohttp.ClientTimeout(total=30)) as file_response:
                                                                    if file_response.status == 200:
                                                                        file_content = await file_response.read()
                                                                        
                                                                        temp_filename = f"{scene_id}_fallback.mp3"
                                                                        temp_filepath = os.path.join(self.task_temp_dir, temp_filename)
                                                                        
                                                                        with open(temp_filepath, "wb") as f:
                                                                            f.write(file_content)
                                                                        
                                                                        if os.path.exists(temp_filepath) and os.path.getsize(temp_filepath) > 0:
                                                                            print(f"✓ HTTP: Fallback successful for scene {scene_id}")
                                                                            return temp_filepath
                                                    except json.JSONDecodeError:
                                                        pass
                                                break
                                            elif "event: error" in event_data:
                                                break
                                            else:
                                                await asyncio.sleep(1)
                                                continue
                                    break
                                
            except Exception as e:
                print(f"✗ HTTP: Final fallback failed for scene {scene_id}: {e}")
        
        print(f"✗ HTTP: All attempts failed for scene {scene_id}")
        return None

    async def generate_all_audio_http(self, scenes: List, voice: str) -> Dict[str, Optional[str]]:
        """Generate audio for all scenes using HTTP API with concurrent requests"""
        scene_audio_map = {}
        
        # Create aiohttp session with connection limits
        connector = aiohttp.TCPConnector(limit=5, limit_per_host=3)  # More conservative limits
        timeout = aiohttp.ClientTimeout(total=600)  # 10 minute total timeout
        
        async with aiohttp.ClientSession(connector=connector, timeout=timeout) as session:
            # Process scenes in smaller batches to avoid overwhelming the API
            batch_size = 3  # Process 3 scenes at a time
            
            for i in range(0, len(scenes), batch_size):
                batch_scenes = scenes[i:i + batch_size]
                
                # Create tasks for this batch
                batch_tasks = []
                for scene in batch_scenes:
                    task = self.generate_single_audio_http(session, scene.text, voice, scene.emotion, scene.id)
                    batch_tasks.append((scene.id, task))
                
                # Process batch concurrently
                for scene_id, task in batch_tasks:
                    try:
                        result = await task
                        scene_audio_map[scene_id] = result
                    except Exception as e:
                        print(f"HTTP: Scene {scene_id} failed with exception: {e}")
                        scene_audio_map[scene_id] = None
                
                # Small delay between batches
                if i + batch_size < len(scenes):
                    await asyncio.sleep(1)
        
        return scene_audio_map


    
    def generate_single_audio(self, text: str, voice: str, emotion: str, scene_id: str) -> Optional[str]:
        """Generate audio for a single scene with enhanced error handling"""
        max_retries = 3
        retry_delay = 1  # seconds
        
        for attempt in range(max_retries):
            try:
                # Clean the text before sending
                cleaned_text = self.clean_text(text)
                print(f"Generating audio for scene {scene_id} (attempt {attempt + 1}/{max_retries}) with text: {cleaned_text[:50]}...")
                
                # Add small delay between requests to avoid rate limiting
                if attempt > 0:
                    import time
                    time.sleep(retry_delay * attempt)
                
                result = self.gradio_client.predict(
                    prompt=cleaned_text,
                    voice=voice,
                    emotion=emotion,
                    use_random_seed=True,
                    specific_seed=12345,
                    api_name="/text_to_speech_app"
                )
                
                # Handle tuple result from Gradio client
                if isinstance(result, (tuple, list)):
                    file_path = result[0] if len(result) > 0 else None
                else:
                    file_path = result
                
                # Verify file exists and has content
                if file_path and os.path.exists(file_path) and os.path.getsize(file_path) > 0:
                    print(f"✓ Generated audio for scene {scene_id}: {file_path}")
                    return file_path
                else:
                    print(f"✗ Invalid/empty audio file for scene {scene_id}: {file_path}")
                    if attempt < max_retries - 1:
                        continue
                
            except Exception as e:
                print(f"✗ Failed to generate audio for scene {scene_id} with emotion '{emotion}' (attempt {attempt + 1}): {e}")
                if attempt < max_retries - 1:
                    continue
                
                # Try with null emotion as final fallback
                print(f"Trying scene {scene_id} with null emotion as final fallback...")
                try:
                    cleaned_text = self.clean_text(text)
                    
                    result = self.gradio_client.predict(
                        prompt=cleaned_text,
                        voice=voice,
                        emotion="",  # null emotion
                        use_random_seed=True,
                        specific_seed=12345,
                        api_name="/text_to_speech_app"
                    )
                    
                    # Handle tuple result from Gradio client
                    if isinstance(result, (tuple, list)):
                        file_path = result[0] if len(result) > 0 else None
                    else:
                        file_path = result
                    
                    # Verify file exists and has content
                    if file_path and os.path.exists(file_path) and os.path.getsize(file_path) > 0:
                        print(f"✓ Retry successful for scene {scene_id} with null emotion: {file_path}")
                        return file_path
                    else:
                        print(f"✗ Null emotion retry also failed for scene {scene_id}: {file_path}")
                        
                except Exception as retry_error:
                    print(f"✗ Final retry failed for scene {scene_id}: {retry_error}")
        
        print(f"✗ All attempts failed for scene {scene_id}")
        return None

    def combine_audio_files(self, audio_files: List[str], output_path: str) -> bool:
        """Combine multiple audio files into a single file"""
        try:
            combined = AudioSegment.empty()
            for audio_file in audio_files:
                # Handle both string paths and tuples (gradio client might return tuples)
                file_path = audio_file
                if isinstance(audio_file, (tuple, list)):
                    file_path = audio_file[0]  # Take first element if it's a tuple
                
                if file_path and os.path.exists(file_path):
                    print(f"Adding audio file to combination: {file_path}")
                    audio = AudioSegment.from_file(file_path)
                    combined += audio
                    # Add small pause between scenes
                    combined += AudioSegment.silent(duration=500)  # 500ms pause
                else:
                    print(f"Audio file not found or invalid: {file_path}")
            
            if len(combined) > 0:
                combined.export(output_path, format="mp3")
                print(f"Successfully combined audio to: {output_path}")
                return True
            else:
                print("No valid audio content to combine")
                return False
        except Exception as e:
            print(f"Failed to combine audio files: {e}")
            print(f"Audio files received: {audio_files}")
            return False
    
    def cleanup_temp_files(self):
        """Clean up all temporary files for this task"""
        try:
            if os.path.exists(self.task_temp_dir):
                shutil.rmtree(self.task_temp_dir)
                print(f"Cleaned up temp directory: {self.task_temp_dir}")
        except Exception as e:
            print(f"Failed to cleanup temp directory {self.task_temp_dir}: {e}")

def update_task_status(task_id: str, status: TaskStatus, progress: str = "", 
                      download_url: str = None, error_message: str = None):
    """Update task status in storage"""
    with tasks_lock:
        if task_id in tasks_storage:
            tasks_storage[task_id].update({
                "status": status,
                "progress": progress,
                "download_url": download_url,
                "error_message": error_message,
                "updated_at": datetime.now()
            })


async def process_tts_task(task_id: str, request: TTSRequest):
    """Process TTS task asynchronously with method selection"""
    processor = None
    try:
        update_task_status(task_id, TaskStatus.PROCESSING, f"Starting audio generation using {request.method}...")
        
        processor = TTSProcessor(task_id)
        
        if request.method == "http_api":
            # Use new HTTP API method
            update_task_status(task_id, TaskStatus.PROCESSING, f"🌐 HTTP: Processing {len(request.scenes)} scenes in parallel...")
            
            scene_audio_map = await processor.generate_all_audio_http(request.scenes, request.voice)
            
            # Log statistics
            successful_scenes = sum(1 for filepath in scene_audio_map.values() if filepath is not None)
            failed_scenes = [scene_id for scene_id, filepath in scene_audio_map.items() if filepath is None]
            total_scenes = len(request.scenes)
            
            print(f"🌐 HTTP: Generation summary: {successful_scenes}/{total_scenes} scenes successful")
            if failed_scenes:
                print(f"🌐 HTTP: Failed scenes: {failed_scenes}")
            
            if successful_scenes == 0:
                error_msg = f"HTTP method: No audio files were generated successfully. Failed scenes: {failed_scenes}"
                update_task_status(task_id, TaskStatus.FAILED, error_message=error_msg)
                return
            
            # Combine audio files maintaining order
            if successful_scenes < total_scenes:
                progress_msg = f"🌐 HTTP: Combining {successful_scenes}/{total_scenes} successful audio files (skipping {len(failed_scenes)} failed scenes)..."
            else:
                progress_msg = "🌐 HTTP: Combining all audio files..."
                
            update_task_status(task_id, TaskStatus.PROCESSING, progress_msg)
            
            # Combine audio files
            output_filename = f"{task_id}.mp3"
            output_path = os.path.join(OUTPUT_DIR, output_filename)
            
            success = processor.combine_audio_files_ordered(scene_audio_map, request.scenes, output_path)
            
            if success:
                download_url = f"/download/{output_filename}"
                final_msg = f"🌐 HTTP: Audio generation completed! Combined {successful_scenes}/{total_scenes} scenes"
                if len(failed_scenes) > 0:
                    final_msg += f" (skipped {len(failed_scenes)} failed scenes)"
                update_task_status(task_id, TaskStatus.COMPLETED, final_msg, download_url)
            else:
                update_task_status(task_id, TaskStatus.FAILED, error_message="HTTP method: Failed to combine audio files")
            
        else:
            # Use original Gradio client method
            audio_files = []
            
            # Process scenes concurrently
            loop = asyncio.get_event_loop()
            
            async def process_scene(scene: Scene):
                return await loop.run_in_executor(
                    executor,
                    processor.generate_single_audio,
                    scene.text,
                    request.voice,
                    scene.emotion,
                    scene.id
                )
            
            update_task_status(task_id, TaskStatus.PROCESSING, f"🔧 Client: Generating audio for {len(request.scenes)} scenes...")
            
            # Create tasks for all scenes
            scene_tasks = [process_scene(scene) for scene in request.scenes]
            
            # Process scenes in batches to avoid overwhelming the API
            batch_size = MAX_CONCURRENT_GRADIO_REQUESTS
            scene_results = []
            
            for i in range(0, len(scene_tasks), batch_size):
                batch = scene_tasks[i:i + batch_size]
                batch_results = await asyncio.gather(*batch, return_exceptions=True)
                scene_results.extend(batch_results)
                
                progress = f"🔧 Client: Processed {min(i + batch_size, len(scene_tasks))}/{len(scene_tasks)} scenes"
                update_task_status(task_id, TaskStatus.PROCESSING, progress)
            
            # Filter successful results and log statistics
            valid_audio_files = []
            failed_scenes = []
            for i, result in enumerate(scene_results):
                if isinstance(result, Exception):
                    print(f"🔧 Client: Scene {request.scenes[i].id} failed with exception: {result}")
                    failed_scenes.append(request.scenes[i].id)
                    continue
                if result:
                    valid_audio_files.append(result)
                else:
                    print(f"🔧 Client: Scene {request.scenes[i].id} returned None")
                    failed_scenes.append(request.scenes[i].id)
            
            # Log processing statistics
            total_scenes = len(request.scenes)
            successful_scenes = len(valid_audio_files)
            failed_count = len(failed_scenes)
            
            print(f"🔧 Client: Audio generation summary: {successful_scenes}/{total_scenes} scenes successful")
            if failed_scenes:
                print(f"🔧 Client: Failed scenes: {failed_scenes}")
            
            if not valid_audio_files:
                error_msg = f"Gradio client method: No audio files were generated successfully. Failed scenes: {failed_scenes}"
                update_task_status(task_id, TaskStatus.FAILED, error_message=error_msg)
                return
            
            # Continue with partial success if we have some audio files
            if failed_count > 0:
                progress_msg = f"🔧 Client: Combining {successful_scenes}/{total_scenes} successful audio files (skipping {failed_count} failed scenes)..."
            else:
                progress_msg = "🔧 Client: Combining audio files..."
                
            update_task_status(task_id, TaskStatus.PROCESSING, progress_msg)
            
            # Combine audio files
            output_filename = f"{task_id}.mp3"
            output_path = os.path.join(OUTPUT_DIR, output_filename)
            
            success = await loop.run_in_executor(
                executor,
                processor.combine_audio_files,
                valid_audio_files,
                output_path
            )
            
            if success:
                download_url = f"/download/{output_filename}"
                final_msg = f"🔧 Client: Audio generation completed successfully! Combined {successful_scenes}/{total_scenes} scenes"
                if failed_count > 0:
                    final_msg += f" (skipped {failed_count} failed scenes)"
                update_task_status(task_id, TaskStatus.COMPLETED, final_msg, download_url)
            else:
                update_task_status(task_id, TaskStatus.FAILED, error_message="Gradio client method: Failed to combine audio files")
            
    except Exception as e:
        print(f"Task {task_id} failed with error: {e}")
        update_task_status(task_id, TaskStatus.FAILED, error_message=str(e))
    finally:
        # Always cleanup temp files, but don't overwrite the final status
        if processor:
            print(f"🧹 Cleaning up temporary files for task {task_id}...")
            processor.cleanup_temp_files()


async def process_tts_task(task_id: str, request: TTSRequest):
    """Process TTS task asynchronously with method selection"""
    processor = None
    try:
        update_task_status(task_id, TaskStatus.PROCESSING, f"Starting audio generation using {request.method}...")
        
        processor = TTSProcessor(task_id)
        
        if request.method == "http_api":
            # Use new HTTP API method
            update_task_status(task_id, TaskStatus.PROCESSING, f"🌐 HTTP: Processing {len(request.scenes)} scenes in parallel...")
            
            scene_audio_map = await processor.generate_all_audio_http(request.scenes, request.voice)
            
            # Log statistics
            successful_scenes = sum(1 for filepath in scene_audio_map.values() if filepath is not None)
            failed_scenes = [scene_id for scene_id, filepath in scene_audio_map.items() if filepath is None]
            total_scenes = len(request.scenes)
            
            print(f"🌐 HTTP: Generation summary: {successful_scenes}/{total_scenes} scenes successful")
            if failed_scenes:
                print(f"🌐 HTTP: Failed scenes: {failed_scenes}")
            
            if successful_scenes == 0:
                error_msg = f"HTTP method: No audio files were generated successfully. Failed scenes: {failed_scenes}"
                update_task_status(task_id, TaskStatus.FAILED, error_message=error_msg)
                return
            
            # Combine audio files maintaining order
            if successful_scenes < total_scenes:
                progress_msg = f"🌐 HTTP: Combining {successful_scenes}/{total_scenes} successful audio files (skipping {len(failed_scenes)} failed scenes)..."
            else:
                progress_msg = "🌐 HTTP: Combining all audio files..."
                
            update_task_status(task_id, TaskStatus.PROCESSING, progress_msg)
            
            # Combine audio files
            output_filename = f"{task_id}.mp3"
            output_path = os.path.join(OUTPUT_DIR, output_filename)
            
            success = processor.combine_audio_files_ordered(scene_audio_map, request.scenes, output_path)
            
            if success:
                download_url = f"/download/{output_filename}"
                final_msg = f"🌐 HTTP: Audio generation completed! Combined {successful_scenes}/{total_scenes} scenes"
                if len(failed_scenes) > 0:
                    final_msg += f" (skipped {len(failed_scenes)} failed scenes)"
                update_task_status(task_id, TaskStatus.COMPLETED, final_msg, download_url)
            else:
                update_task_status(task_id, TaskStatus.FAILED, error_message="HTTP method: Failed to combine audio files")
            
        else:
            # Use original Gradio client method
            audio_files = []
            
            # Process scenes concurrently
            loop = asyncio.get_event_loop()
            
            async def process_scene(scene: Scene):
                return await loop.run_in_executor(
                    executor,
                    processor.generate_single_audio,
                    scene.text,
                    request.voice,
                    scene.emotion,
                    scene.id
                )
            
            update_task_status(task_id, TaskStatus.PROCESSING, f"🔧 Client: Generating audio for {len(request.scenes)} scenes...")
            
            # Create tasks for all scenes
            scene_tasks = [process_scene(scene) for scene in request.scenes]
            
            # Process scenes in batches to avoid overwhelming the API
            batch_size = MAX_CONCURRENT_GRADIO_REQUESTS
            scene_results = []
            
            for i in range(0, len(scene_tasks), batch_size):
                batch = scene_tasks[i:i + batch_size]
                batch_results = await asyncio.gather(*batch, return_exceptions=True)
                scene_results.extend(batch_results)
                
                progress = f"🔧 Client: Processed {min(i + batch_size, len(scene_tasks))}/{len(scene_tasks)} scenes"
                update_task_status(task_id, TaskStatus.PROCESSING, progress)
            
            # Filter successful results and log statistics
            valid_audio_files = []
            failed_scenes = []
            for i, result in enumerate(scene_results):
                if isinstance(result, Exception):
                    print(f"🔧 Client: Scene {request.scenes[i].id} failed with exception: {result}")
                    failed_scenes.append(request.scenes[i].id)
                    continue
                if result:
                    valid_audio_files.append(result)
                else:
                    print(f"🔧 Client: Scene {request.scenes[i].id} returned None")
                    failed_scenes.append(request.scenes[i].id)
            
            # Log processing statistics
            total_scenes = len(request.scenes)
            successful_scenes = len(valid_audio_files)
            failed_count = len(failed_scenes)
            
            print(f"🔧 Client: Audio generation summary: {successful_scenes}/{total_scenes} scenes successful")
            if failed_scenes:
                print(f"🔧 Client: Failed scenes: {failed_scenes}")
            
            if not valid_audio_files:
                error_msg = f"Gradio client method: No audio files were generated successfully. Failed scenes: {failed_scenes}"
                update_task_status(task_id, TaskStatus.FAILED, error_message=error_msg)
                return
            
            # Continue with partial success if we have some audio files
            if failed_count > 0:
                progress_msg = f"🔧 Client: Combining {successful_scenes}/{total_scenes} successful audio files (skipping {failed_count} failed scenes)..."
            else:
                progress_msg = "🔧 Client: Combining audio files..."
                
            update_task_status(task_id, TaskStatus.PROCESSING, progress_msg)
            
            # Combine audio files
            output_filename = f"{task_id}.mp3"
            output_path = os.path.join(OUTPUT_DIR, output_filename)
            
            success = await loop.run_in_executor(
                executor,
                processor.combine_audio_files,
                valid_audio_files,
                output_path
            )
            
            if success:
                download_url = f"/download/{output_filename}"
                final_msg = f"🔧 Client: Audio generation completed successfully! Combined {successful_scenes}/{total_scenes} scenes"
                if failed_count > 0:
                    final_msg += f" (skipped {failed_count} failed scenes)"
                update_task_status(task_id, TaskStatus.COMPLETED, final_msg, download_url)
            else:
                update_task_status(task_id, TaskStatus.FAILED, error_message="Gradio client method: Failed to combine audio files")
            
    except Exception as e:
        print(f"Task {task_id} failed with error: {e}")
        update_task_status(task_id, TaskStatus.FAILED, error_message=str(e))
    finally:
        # Always cleanup temp files, but don't overwrite the final status
        if processor:
            print(f"🧹 Cleaning up temporary files for task {task_id}...")
            processor.cleanup_temp_files()



@app.post("/generate-audio", response_model=TaskResponse)
async def generate_audio(request: TTSRequest, background_tasks: BackgroundTasks):
    """Submit a TTS generation task"""
    if not request.scenes:
        raise HTTPException(status_code=400, detail="No scenes provided")
    
    task_id = str(uuid.uuid4())
    
    # Store task info
    with tasks_lock:
        tasks_storage[task_id] = {
            "status": TaskStatus.PENDING,
            "progress": "Task submitted",
            "download_url": None,
            "error_message": None,
            "created_at": datetime.now(),
            "updated_at": datetime.now(),
            "request": request.dict()
        }
    
    # Start background processing
    background_tasks.add_task(process_tts_task, task_id, request)
    
    return TaskResponse(
        task_id=task_id,
        status=TaskStatus.PENDING,
        message="Task submitted successfully"
    )

@app.get("/task_status/{task_id}", response_model=TaskStatusResponse)
async def get_task_status(task_id: str):
    """Get the status of a TTS generation task"""
    with tasks_lock:
        if task_id not in tasks_storage:
            raise HTTPException(status_code=404, detail="Task not found")
        
        task_info = tasks_storage[task_id]
    
    return TaskStatusResponse(
        task_id=task_id,
        status=task_info["status"],
        progress=task_info["progress"],
        download_url=task_info["download_url"],
        error_message=task_info["error_message"]
    )

@app.get("/download/{filename}")
async def download_audio(filename: str):
    """Download generated audio file"""
    file_path = os.path.join(OUTPUT_DIR, filename)
    
    if not os.path.exists(file_path):
        raise HTTPException(status_code=404, detail="File not found")
    
    from fastapi.responses import FileResponse
    return FileResponse(
        file_path,
        media_type="audio/mpeg",
        filename=filename
    )

@app.get("/tasks")
async def list_tasks():
    """List all tasks (for debugging)"""
    with tasks_lock:
        return {
            task_id: {
                "status": task_info["status"],
                "progress": task_info["progress"],
                "created_at": task_info["created_at"].isoformat(),
                "updated_at": task_info["updated_at"].isoformat()
            }
            for task_id, task_info in tasks_storage.items()
        }

@app.delete("/tasks/{task_id}")
async def delete_task(task_id: str):
    """Delete a task and its associated files"""
    with tasks_lock:
        if task_id not in tasks_storage:
            raise HTTPException(status_code=404, detail="Task not found")
        
        task_info = tasks_storage[task_id]
        
        # Delete associated audio file if exists
        if task_info["download_url"]:
            filename = task_info["download_url"].split("/")[-1]
            file_path = os.path.join(OUTPUT_DIR, filename)
            try:
                if os.path.exists(file_path):
                    os.remove(file_path)
            except:
                pass
        
        # Clean up any remaining temp files for this task
        task_temp_dir = os.path.join(TEMP_DIR, task_id)
        try:
            if os.path.exists(task_temp_dir):
                shutil.rmtree(task_temp_dir)
        except:
            pass
        
        # Remove from storage
        del tasks_storage[task_id]
    
    return {"message": "Task deleted successfully"}

@app.get("/cleanup_temp")
async def cleanup_temp():
    """Cleanup all temporary files (admin endpoint)"""
    try:
        if os.path.exists(TEMP_DIR):
            for item in os.listdir(TEMP_DIR):
                item_path = os.path.join(TEMP_DIR, item)
                if os.path.isdir(item_path):
                    shutil.rmtree(item_path)
                else:
                    os.remove(item_path)
        return {"message": "Temporary files cleaned up successfully"}
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Failed to cleanup temp files: {str(e)}")

@app.get("/")
async def root():
    return {
        "message": "TTS Audio Generator API",
        "endpoints": {
            "generate_audio": "POST /generate-audio",
            "task_status": "GET /task-status/{task_id}",
            "download": "GET /download/{filename}",
            "list_tasks": "GET /tasks",
            "cleanup_temp": "GET /cleanup-temp"
        }
    }

if __name__ == "__main__":
    uvicorn.run(app, host="0.0.0.0", port=5002)