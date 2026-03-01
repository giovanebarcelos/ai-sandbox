# GO1262-CarregarFashionMnist
# Carregar Fashion MNIST


if __name__ == "__main__":
    (x_train, y_train), (x_test, y_test) = \
        keras.datasets.fashion_mnist.load_data()

    # Classes
    class_names = ['T-shirt', 'Trouser', 'Pullover', 'Dress', 
                   'Coat', 'Sandal', 'Shirt', 'Sneaker', 
                   'Bag', 'Ankle boot']
