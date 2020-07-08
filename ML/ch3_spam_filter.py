import tarfile
import bz2

filepath = "C:\\Users\\tamas\\OneDrive\\python_code\\ML\\datasets\\spam_ham\\20021010_easy_ham.tar.bz2"
zipfile = bz2.BZ2File("C:\\Users\\tamas\\OneDrive\\python_code\\ML\\datasets\\spam_ham\\20021010_easy_ham.tar.bz2")
data = zipfile.read()
newfilepath = filepath[:-4]
open(newfilepath, 'wb').write(data)
