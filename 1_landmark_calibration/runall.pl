@au = (surprise);
system("mkdir cmt");
for($i=0;$i<scalar(@au);$i++){

  $dirname = $au[$i];

   print("cmt/".$dirname."\n");
   system("mkdir cmt/".$dirname);  

   opendir(DIR, "kaggle_sub_train/".$dirname) or die "cannot open directory";
   @docs = grep(/\.png/,readdir(DIR));
   foreach $file (@docs) {
    ($name)=$file=~/(.*?)\.png/;
    #print("python facial_landmarks.py -p shape_predictor_68_face_landmarks.dat -i kaggle_sub_train/$dirname/$file > cmt/$dirname/$name.txt \n");
    system("python facial_landmarks.py -p shape_predictor_68_face_landmarks.dat -i kaggle_sub_train/$dirname/$file > cmt/$dirname/$name.txt");
   }
   close(DIR);

}
