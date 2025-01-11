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