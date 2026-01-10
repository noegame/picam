def hello():
    """Import file hello.txt and print its content."""
    with open("vision_python/src/hello/hello.txt", "r") as file:
        content = file.read()
        print(content)
