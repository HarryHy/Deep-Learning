import ConfigParser
config = ConfigParser.ConfigParser()
file = open("config.ini", "w")
config.add_section('Person')
config.set('Person', 'Name', "Todd")
config.set('Person', 'Age', '19')
config.write(file)
config
file.close()