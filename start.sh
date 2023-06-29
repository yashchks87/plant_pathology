source ../tensor_env/bin/activate

mkdir ./files/
export KAGGLE_USERNAME=yashchoksi16
export KAGGLE_KEY=961ee48f863626a69b28671a84d21e7a

kaggle competitions download -c plant-pathology-2021-fgvc8

mv *.zip ./files/

unzip ./files/*.zip -d ./files/