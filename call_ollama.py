import subprocess


def run_ollama_command(model, prompt):
    try:
        # Execute the ollama command using subprocess
        result = subprocess.run(
            ["ollama", "run", model],
            input=prompt,
            text=True,
            capture_output=True,
            check=True,
        )

        # Output the result from Ollama
        print("Response from Ollama:")
        print(result.stdout)
        return result.stdout

    except subprocess.CalledProcessError as e:
        # Handle errors in case of a problem with the command
        print("Error executing Ollama command:")
        print(e.stderr)


# # Example usage
# if __name__ == "__main__":
#     model_name = "llama2"  # Replace with the specific model you want to use
#     user_prompt = "Hello, how are you today?"
#
#     run_ollama_command(model_name, user_prompt)
