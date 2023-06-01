#### Bundling the source

Running Python code requires an interpreter and specific external libraries. That is not always
convenient, so [PyInstaller](https://pyinstaller.org/en/stable/) allows bundling the interpreter and modules in one singular file or folder.
The source code can be found in the `src` directory. To make an installer [Inno Setup Compiler](https://jrsoftware.org/isdl.php)
was used.<br />

#### Instructions

Python 3.10 version was used. Required packages can be installed using the pip command:

```pip install -r requirements.txt```

To build run:

```build.bat```

Edit build.bat to change the build version.