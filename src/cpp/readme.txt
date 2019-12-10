Author: Guanfu Liu

To run the code, need to do the following:
1. This project relies on c++ 17, so need to configure your g++ to latest version (we used gcc-8.1 on GHC by putting this line in MakeFile: Use CXX = /usr/local/depot/gcc-8.1/bin/g++)
2. Put this line in your bashrc file: export LD_LIBRARY_PATH=/usr/local/depot/gcc-8.1/lib64/:${LD_LIBRARY_PATH}
3. Need to download the dataset from LibSVM with the required form and put them in data/ folder. We used the following 2 datasets:
        1. E2006-tfidf, https://www.csie.ntu.edu.tw/~cjlin/libsvmtools/datasets/regression/E2006.train.bz2
        2. cadata.txt, https://www.csie.ntu.edu.tw/~cjlin/libsvmtools/datasets/regression/cadata
4. Simply run make and run the test.sh bash file attached in the cpp/ folder. Feel free to modify hyper parameters in the test.sh file.
