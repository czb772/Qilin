split=$1
language=$2

if [ $language == "en" ]; then
    index="../datasets/qilin_notes_en_index"
else
    index="../datasets/qilin_notes_index"
fi

if [ $split == "dqa" ]; then
    java -cp $ANSERINI_JAR io.anserini.search.SearchCollection -index $index -topics ../datasets/qilin_queries/dqa_queries.txt -topicReader TsvString -output ../datasets/qilin_bm25_runs/dqa_$language.txt -bm25 -hits 100 -threads 100 -language $language
elif [ $split == "search_test" ]; then
    java -cp $ANSERINI_JAR io.anserini.search.SearchCollection -index $index -topics ../datasets/qilin_queries/search_test_queries.txt -topicReader TsvString -output ../datasets/qilin_bm25_runs/search_test_$language.txt -bm25 -hits 100 -threads 100 -language $language
elif [ $split == "search_train" ]; then
    java -cp $ANSERINI_JAR io.anserini.search.SearchCollection -index $index -topics ../datasets/qilin_queries/search_train_queries.txt -topicReader TsvString -output ../datasets/qilin_bm25_runs/search_train_$language.txt -bm25 -hits 100 -threads 100 -language $language
elif [ $split == "recommendation_train" ]; then
    java -cp $ANSERINI_JAR io.anserini.search.SearchCollection -index $index -topics ../datasets/qilin_queries/recommendation_train_queries.txt -topicReader TsvString -output ../datasets/qilin_bm25_runs/recommendation_train_$language.txt -bm25 -hits 100 -threads 100 -language $language
elif [ $split == "recommendation_test" ]; then
    java -cp $ANSERINI_JAR io.anserini.search.SearchCollection -index $index -topics ../datasets/qilin_queries/recommendation_test_queries.txt -topicReader TsvString -output ../datasets/qilin_bm25_runs/recommendation_test_$language.txt -bm25 -hits 100 -threads 100 -language $language
fi