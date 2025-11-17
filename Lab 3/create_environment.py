from azureml.core import Workspace, Environment

# Load workspace
ws = Workspace.from_config()

# Create environment from conda file
env = Environment.from_conda_specification(
    name='wine-quality-env',
    file_path='environment.yml'
)

# Register the environment
env.register(workspace=ws)

print(f"Environment '{env.name}' registered successfully!")
print(f"Environment version: {env.version}")