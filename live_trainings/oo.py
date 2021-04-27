class Bike:

    def __init__(self, price=None):
        if price is None:
            self.price = 0
        else:
            self.price = price

        self.age = 1

    def __repr__(self):
        return 'This is my bike'

    def update_sale_price(self, change=0):
        self.price = self.price + change

    def service(self, change=0):
        self.age += change

if __name__ == '__main__':
    mybike = Bike()
    mybike.update_sale_price(100)
    mybike.price
    myotherbike = Bike(100)
    mybike.service(10)
    mybike.age
