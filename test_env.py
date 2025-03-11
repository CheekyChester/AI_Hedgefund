import sys
print("Python version:", sys.version)
print("Python executable:", sys.executable)
print("Path:", sys.path)

try:
    import langchain
    print("langchain version:", langchain.__version__)
except ImportError:
    print("langchain not found")

try:
    import dotenv
    print("python-dotenv version:", dotenv.__version__)
except ImportError:
    print("python-dotenv not found")

try:
    import jinja2
    print("jinja2 version:", jinja2.__version__)
except ImportError:
    print("jinja2 not found")