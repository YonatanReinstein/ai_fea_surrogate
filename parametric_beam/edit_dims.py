from pathlib import Path

def write_irit_params(file_path, params):
    lines = [f"{key} = {value};" for key, value in params.items()]
    content = "\n".join(lines) + "\n"
    Path(file_path).write_text(content)

if __name__ == "__main__":
    params = {
        "Length": 12,
        "Width": 6,
        "Height": 8,
    }
    write_irit_params("dims.irt", params)
