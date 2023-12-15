Individual assignment Product Duplicate Detection - Computer Science for Business Analytics

Programming partner: Isa van Stee (574699is)

This project is about reducing the computation time of a duplicate detection method by using LSH. LSH can reduce the number of pairs which have to be considered by the classification (or clustering) method. We apply the algorithm to a dataset containing TV's from 4 different Web shop and some of the TV's are duplicates. The dataset contains in total 1624 TV's, of which 1262 are unique products. We use the title and the values of the key-value pairs to obtain a binary representation of a product. We do not consider all key-value pairs but only the top 10 most occurring key-value pairs in the data.

The code is structured as follows:

First, we read the data and obtain the top 10 most occurring key-value pairs. After that, we clean the titles of the products by using the method proposed by [1]. Next, we obtain model words from the cleaned titles and obtain the values from the key-value pairs. We use these to create binary vectors representing the products. Putting these vectors together results in a binary matrix. After that, we use the minhashing algorithm to create a signature matrix which will then be used as the input matrix in the LSH algorithm. The LSH algortihm provides a list of candidate pairs and we use a classifying method to find the duplicates. For the classification algorithm, we use the Jaccard similarity. Lastly, we obtain the performance metrics Pair Completeness (PC), Pair Quality (PQ),  F1-measure and F1*-measure.

To ensure robustness, we perform 5 boostraps in total and average the results over the 5 bootstraps. We consider multiple combinations of bands and rows in the LSH algorithm. This is done so that we can observe the effect of the size of the fraction of comparisons on the different performance measures. For the minhashing algorithm, we have set the number of permutations at 600, as this is approximately 50% of the size of the binary vector. We have set the Jaccard similarity in the classification method manually at 0.8.

Running the code provides the plots which are discussed in the paper and the average performance metrics over the 5 bootstraps.


[1] Hartveld, A., van Keulen, M., Mathol, D., van Noort, T., Plaatsman, T., Frasin-
car, F., Schouten, K.: An lsh-based model-words-driven product duplicate detection
method. In: International Conference on Advanced Information Systems Engineer-
ing. pp. 409–423. Springer (2018)
