echo "Start downloading embeddings.."
curl -0 "https://conceptnet.s3.amazonaws.com/downloads/2017/numberbatch/numberbatch-17.06.txt.gz" 
echo "Embeddings downloaded. Start unzipping."
unzip numberbatch-17.06.txt.gz
echo "Embeddings unzipped. Start filtering English and Spanish words."
python extract_embeddings.py
echo "Done. Cleaning.."
rm numberbatch-17.06.txt
echo "Done!"
echo 
echo
echo 

