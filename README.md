# ARTag Localizer

Código responsável por ler stream da webcam, processar imagens para detectar ARTags e retornar informações posicionais.

## Estrutura do Projeto
ARTag-localizer/ ├── pycache/ ├── .gitignore ├── detectGrid.py ├── h_mat.npy ├── inv_h.npy ├── main.py ├── out.json ├── README.md ├── requirements.txt ├── server.py ├── testCapture.py ├── utils/ │ ├── pycache/ │ ├── detector.py │ ├── draw_3d.py │ ├── setupTransform.py │ ├── superimpose.py └── venv/ ├── .gitignore ├── bin/ ├── include/ ├── lib/ ├── lib64 ├── pyvenv.cfg └── share/


## Requisitos

Para instalar as dependências necessárias, execute:

```sh
pip install -r requirements.txt


## Uso
Para executar o script principal, execute:
python main.py


Files
main.py: Main script to process the webcam stream and detect ARTags.
detectGrid.py: Script to detect the grid in the image.
server.py: Script to run the server.
testCapture.py: Script to capture a single image from the webcam.
utils: Directory containing utility scripts for detection, drawing, and transformations.
h_mat.npy and inv_h.npy: Numpy files storing the homography matrices.
out.json: JSON file storing the output positional information.


## License

This project is licensed under the MIT License. ```


