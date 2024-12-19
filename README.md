# Brain Tumor Detection

This project is a machine learning application designed to detect brain tumors based on user input MRI images. The model classifies the images into one of four categories:

1. No Tumor (Healthy)
2. Gliomas
3. Pituitary
4. Meningiomas

## Table of Contents

- [Installation](#installation)
- [Usage](#usage)
- [Model](#model)
- [Contributing](#contributing)
- [License](#license)

## Installation

To install the necessary dependencies, run the following command:

```bash
pip install -r requirements.txt
```

## Usage

To use the model for brain tumor detection, follow these steps:

1. Provide an MRI image as input.
2. Run the model to classify the image.
3. The model will output one of the four classes.

Example command:

```bash
python model.py
```

## Model

The model is trained on a dataset of MRI images and uses deep learning techniques to classify the images. The four classes are:

- **No Tumor (Healthy)**: Indicates that the MRI image shows no signs of a tumor.
- **Gliomas**: A type of tumor that occurs in the brain and spinal cord.
- **Pituitary**: Tumors that occur in the pituitary gland.
- **Meningiomas**: Tumors that arise from the meninges, the membranes that surround the brain and spinal cord.

## Contributing

Contributions are welcome! Please fork the repository and submit a pull request.

## License

This project is licensed under the MIT License. See the [LICENSE](LICENSE) file for details.