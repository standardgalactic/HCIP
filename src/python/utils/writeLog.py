
class writeLog:
    def __init__(self, save_path):
        self.filename = str(save_path)+'.log'

    def write(self, text):
        f = open(self.filename, 'a')
        f.write(text)
        f.close()
