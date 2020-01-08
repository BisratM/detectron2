module purge
module load anaconda3
git clone https://github.com/danielleung27/deepmask-pytorch.git
cd deepmask-pytorch
conda create --name deepmask python=3 -y
conda activate deepmask
conda install torchvision -c pytorch -y
conda install opencv -y
conda install matplotlib -y
conda install Cython -y
conda install colorama -y
pip install tensorboardX
pip install "pillow<7"
DEEPMASK=$PWD
export PYTHONPATH=$DEEPMASK:$PYTHONPATH

python -c "import torchvision; torchvision.datasets.VOCSegmentation('pascal_train/', year='2012', image_set='train', download=True)"
rm "./pascal_train/VOCtrainval_11-May-2012.tar"
python -c "import torchvision; torchvision.datasets.VOCSegmentation('pascal_val/', year='2012', image_set='val', download=True)"
rm "./pascal_val/VOCtrainval_11-May-2012.tar"
wget https://storage.googleapis.com/coco-dataset/external/PASCAL_VOC.zip

unzip PASCAL_VOC.zip
rm PASCAL_VOC.zip

mkdir -p $DEEPMASK/data/coco
mkdir -p $DEEPMASK/data/coco/annotations

mv $DEEPMASK/pascal_train/VOCdevkit/VOC2012/JPEGImages/ $DEEPMASK/data/coco
mv $DEEPMASK/data/coco/JPEGImages $DEEPMASK/data/coco/train2017

mv $DEEPMASK/pascal_val/VOCdevkit/VOC2012/JPEGImages/ $DEEPMASK/data/coco
mv $DEEPMASK/data/coco/JPEGImages $DEEPMASK/data/coco/val2017

mv $DEEPMASK/PASCAL_VOC/pascal_train2012.json $DEEPMASK/data/coco/annotations
mv $DEEPMASK/data/coco/annotations/pascal_train2012.json $DEEPMASK/data/coco/annotations/instances_train2017.json

mv $DEEPMASK/PASCAL_VOC/pascal_val2012.json $DEEPMASK/data/coco/annotations
mv $DEEPMASK/data/coco/annotations/pascal_val2012.json $DEEPMASK/data/coco/annotations/instances_val2017.json

rm -r $DEEPMASK/pascal_train
rm -r $DEEPMASK/pascal_val
rm -r $DEEPMASK/PASCAL_VOC

cd $DEEPMASK/loader/pycocotools && make
cd $DEEPMASK

CUDA_VISIBLE_DEVICES=0,1,2,3 python tools/train.py --dataset coco -j 20 --freeze_bn