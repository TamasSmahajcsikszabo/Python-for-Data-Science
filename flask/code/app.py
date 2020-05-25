from flask import Flask, request
from flask_restful import Resource, Api

app = Flask(__name__)
api = Api(app)

items = []

# the APi works with resources, and every resource has to be a class
class Item(Resource):
    def get(self, name):
        item = next(filter(lambda x: x['name'] == name, items), None) #None is the default value
        return {'item': None}, 200 if item is not None else 404

    def post(self, name):
        if next(filter(lambda x: x['name'] == name, items), None) is not None:
            return {'message' : "An item with name '{}' already exists".format(name)}, 400

        data = request.get_json(force=True) #silent=True returns null without error; force=True enforces JSON format
        item = {'name': name, 'price': data['price']}
        items.append(item)
        return item, 201

class ItemList(Resource):
    def get(self):
        return items

api.add_resource(Item, '/item/<string:name>')
api.add_resource(ItemList, '/items')

app.run(port=3333, debug=True)
