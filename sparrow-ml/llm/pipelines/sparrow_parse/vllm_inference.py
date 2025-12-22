from sparrow_parse.vllm.inference_base import ModelInference
from openai import OpenAI
import os
import json
import re
import base64

class vLLMInference(ModelInference):

    def __init__(self, model_name):
        """
        Initialize the inference class with the given model name.

        :param model_name: Name of the model to load.
        """
        self.model_name = model_name
        self.client = OpenAI(base_url="http://localhost:8004/v1", api_key="not-needed")
        print(f"vLLM inference initialized for model: {model_name}")

    def process_response(self, output_text):
        """
        Process and clean the model's raw output to format as JSON.
        """
        try:
            # Check if we have markdown code block markers
            if "```" in output_text:
                # Handle markdown-formatted output
                json_start = output_text.find("```json")
                if json_start != -1:
                    # Extract content between ```json and ```
                    content = output_text[json_start + 7:]
                    json_end = content.rfind("```")
                    if json_end != -1:
                        content = content[:json_end].strip()
                        formatted_json = json.loads(content)
                        return json.dumps(formatted_json, indent=2, ensure_ascii=False)

            # Handle raw JSON (no markdown formatting)
            # First try to find JSON array or object patterns
            for pattern in [r'\[\s*\{.*\}\s*\]', r'\{.*\}']:
                matches = re.search(pattern, output_text, re.DOTALL)
                if matches:
                    potential_json = matches.group(0)
                    try:
                        formatted_json = json.loads(potential_json)
                        return json.dumps(formatted_json, indent=2, ensure_ascii=False)
                    except:
                        pass

            # Last resort: try to parse the whole text as JSON
            formatted_json = json.loads(output_text.strip())
            return json.dumps(formatted_json, indent=2, ensure_ascii=False)

        except Exception as e:
            print(f"Failed to parse JSON: {e}")
            return output_text

    def inference(self, input_data, apply_annotation=False, precision_callback=None, mode=None):
        """
        Perform inference on input data using the specified model.

        :param input_data: A list of dictionaries containing image file paths and text inputs.
        :param apply_annotation: Optional flag to apply annotations to the output.
        :param precision_callback: Optional callback function to modify input data before inference.
        :param mode: Optional mode for inference ("static" for simple JSON output).
        :return: List of processed model responses.
        """

        # Validate input_data
        if not input_data or not isinstance(input_data, list) or len(input_data) == 0:
            raise ValueError("input_data must be a non-empty list")
        
        apply_annotation = False

        file_paths = self._extract_file_paths(input_data)
        results = self._process_images(file_paths, input_data)

        return results


    def _process_images(self, file_paths, input_data):
        """
        Process images and generate responses for each.
        """
        results = []
        for file_path in file_paths:
            try:
                # Check if file exists
                if not os.path.exists(file_path):
                    print(f"Warning: File does not exist: {file_path}")
                    continue

                with open(file_path, "rb") as f:
                    base64_image = base64.b64encode(f.read()).decode("utf-8")

                # Make the multimodal request to TensorRT
                response = self.client.chat.completions.create(
                    model=self.model_name,
                    messages=[
                        {"role": "system", "content": "You are a helpful assitant."},
                        {
                            "role": "user", 
                            "content": [
                                {"type": "text", "text": input_data[0]["text_input"]},
                                {
                                    "type": "image_url",
                                    "image_url": {"url": f"data:image/jpeg;base64,{base64_image}"}
                                },
                            ],
                        }
                    ],
                    max_completion_tokens=256,
                    temperature=0.0,
                    stop=["<|im_end|>", "</s>", "<|endoftext|>"],
                )

                # Process the raw response
                processed_response = self.process_response(response.choices[0].message.content)

                results.append(processed_response)
                print(f"Inference completed successfully for: {file_path}")

            except Exception as e:
                print(f"Error processing image {file_path}: {e}")
                # Continue processing other images instead of failing completely
                continue

        return results

    @staticmethod
    def _extract_file_paths(input_data):
        """
        Extract and resolve absolute file paths from input data.

        :param input_data: List of dictionaries containing image file paths.
        :return: List of absolute file paths.
        """
        return [
            os.path.abspath(file_path)
            for data in input_data
            for file_path in data.get("file_path", [])
        ]
