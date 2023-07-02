Apologies for the oversight. Here's an updated version of the README.md file with a more explicit instruction for contributing:


# ImageCaptionGenerator

ImageCaptionGenerator is a Flask application that utilizes TensorFlow to generate descriptions for images. It provides an API endpoint that accepts a base64-encoded JPEG image and returns a description for the given image.

## Usage

To use this application, you need to communicate with port 5000 of the Docker container. Make a POST request to `http://localhost:5000/ask` with the image data provided in the request body as JSON, like this:

```json
{
  "img": "base64-encoded-image-data"
}
```

The application uses a TensorFlow model trained on the Flickr8K dataset to generate descriptions for images. It leverages deep learning techniques to analyze the provided image and generate a human-readable description. The model is included in the Docker image, so no additional setup is required.

## How to Run

To run the application locally using Docker, follow these steps:

1. Pull the Docker image from Docker Hub:

   ```bash
   docker pull wasinuddy/image-2-text:v1
   ```

2. Run the Docker container:

   ```bash
   docker run -p 5000:5000 wasinuddy/image-2-text:v1
   ```

   This command maps port 5000 of your local machine to port 5000 of the Docker container, allowing you to access the Flask application at `http://localhost:5000/ask`.

## Demo Image and Generated Description

![Demo Image](https://github.com/WasinUddy/ImageCaptionGenerator/blob/main/demo/demo-img.jpg)

Generated Description: mountain biker rides through the forest

<!-- Add more demo images and descriptions if desired -->

## Contributing

Contributions are welcome! If you'd like to contribute to this project, please follow these steps:


## License

This project is licensed under the [MIT License](LICENSE).
