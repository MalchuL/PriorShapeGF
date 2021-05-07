
Instalation
pip install -r requrements.txt

#### Install Kaolin (For PointNet++)
In 0.1 version PointNet exist, they are remove all geometric models from next versions.

git clone --recursive https://github.com/NVIDIAGameWorks/kaolin
cd kaolin
git checkout v0.1

python setup.py develop

Test kaolin import
python -c "import kaolin; print(kaolin.__version__)"

## Dataset
Please follow the instruction from PointFlow to set-up the dataset: (link)[https://drive.google.com/drive/folders/1G0rf-6HSHoTll6aH7voh-dXj6hCRhSAQ].