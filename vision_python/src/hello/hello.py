def hello():
    """Import file hello.txt and print its content."""
    with open("hello.txt", "r") as file:
        content = file.read()
        print(content)
