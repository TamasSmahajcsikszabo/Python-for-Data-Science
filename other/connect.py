from IPython import get_ipython
import json, contextlib, io

ipython = get_ipython()
con = io.StringIO()

with contextlib.redirect_stdout(con):
    ipython.magic("connect_info")
connection = con.getvalue()

end = connection.find("}", 1, len(connection))
con_sub = connection[4:end-1]
con_sub = con_sub.split('\n')

c = {}
for i in range(0, len(con_sub)):
    (key, val) = con_sub[i].strip().split(':')
    key = key.replace("'","")
    val = val.strip().replace(",","")
    c[key] = val
print(c)

with open('/home/tamas/repos/pythonDS/connection.json', 'w') as outfile:
    json.dump(c, outfile)
    print("Connection saved")

# testing
with open('/home/tamas/repos/pythonDS/connection.json') as json_file:
    data = json.load(json_file)
print(data)
