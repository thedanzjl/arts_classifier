
import os

labels = ['drawings', 'engraving', 'iconography', 'painting', 'sculpture']
dataset_folder = 'dataset_updated'


def generate_csv_with_paths(name, path):
    file = open(name, 'w')
    for i, label in enumerate(labels):
        temp_path = os.path.join(path, label)
        images = os.listdir(temp_path)
        for img_name in images:
            file.write('{},{}\n'.format(os.path.join(temp_path, img_name), i))
    file.close()


def generate_test_csv(name, path):
    file = open(name, 'w')
    images = os.listdir(path)
    for image in images:
        file.write(os.path.join(path, image) + '\n')
    file.close()


if __name__ == '__main__':
    generate_csv_with_paths('training.csv', path=os.path.join(dataset_folder, 'training_set'))
    generate_csv_with_paths('validation.csv', path=os.path.join(dataset_folder,'validation_set'))
