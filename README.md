# ImageCaptionGenerator

ImageCaptionGenerator is a Flask application that utilizes TensorFlow to generate descriptions for images. It provides an API endpoint that accepts a base64-encoded JPEG image and returns a description for the given image.

## Usage

To use this application, you need to communicate with port 5000 of the Docker container. Make a POST request to `http://localhost:5000/ask` with the image data provided in the request body as JSON, like this:

```json
{
  "img": "base64-encoded-image-data"
}
```

The application uses a TensorFlow model trained to generate descriptions for images. It leverages deep learning techniques to analyze the provided image and generate a human-readable description. The model is included in the Docker image, so no additional setup is required.

## How to Run

To run the application locally, you need to have Docker installed on your machine. Follow these steps:

1. Clone this repository:

   ```bash
   git clone https://github.com/WasinUddy/ImageCaptionGenerator.git
   cd ImageCaptionGenerator
   ```

2. Build the Docker image:

   ```bash
   docker build -t image-2-text .
   ```

3. Run the Docker container:

   ```bash
   docker run -p 5000:5000 image-2-text
   ```

   This command maps port 5000 of your local machine to port 5000 of the Docker container, allowing you to access the Flask application at `http://localhost:5000/ask`.

## Contributing

Contributions are welcome! If you'd like to contribute to this project, please follow these steps:

1. Fork the repository.

2. Create a new branch.

   ```bash
   git checkout -b my-new-feature
   ```

3. Make your changes and commit them.

   ```bash
   git commit -m 'Add some feature'
   ```

4. Push the changes to your fork.

   ```bash
   git push origin my-new-feature
   ```

5. Submit a pull request.

Please ensure that your code adheres to the existing code style and includes appropriate documentation and tests where necessary.

## License

This project is licensed under the [MIT License](LICENSE).
