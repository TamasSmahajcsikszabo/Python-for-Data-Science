bob = ['Bob Smith', 40.5, 32000, 'software']
sue = ['sue Jones', 44, 54000, 'hardware']

people = [bob, sue]

for person in people:
    print(person[0].split()[-1])
    person[2] *= 1.25
    print(person[2])

pays = [person[2] for person in people]

pays2 = list(map((lambda x: x[2]),people))

pays3 = list((person[2] for person in people))

people.append(['Tom', 34, 0, None])

# associating a name with fields by using range()
NAME, AGE, PAY, SPECS = range(4)
[person[NAME] for person in people]

fields = ["NAME", "AGE", "PAY", "SPECS"]
people2 = []
for i in range(len(people)):
    new_record = []
    for j in range(len(fields)):
        new_record.append([fields[j], people[i][j]])
    people2.append(new_record)


def field(record, label):
    for (fname, fvalue) in record:
        if fname == label:
            return fvalue

for rec in people2:
    print(field(rec, "NAME"))


# dictionaries
people = {}

for person in range(len(people2)):
    new_person = {rec[0]: rec[1] for rec in people2[person]}
    people.update({person: new_person})

# initializing empty dicts
fields = ('name', 'age', 'pay', 'specs')
record = dict.fromkeys(fields, '?')
