using System;
using System.Globalization;
using System.Linq;
using MathNet.Numerics.LinearAlgebra;
using MathNet.Numerics.Data.Text;
using System.Collections.Generic;
using Annytab.Stemmer;
using System.Text;
using MathNet.Numerics;
using libsvm;
using MathNet.Numerics.Statistics;

namespace SentimentAnalysis
{
    class Program
    {
        static void Main(string[] args)
        {
            if(!System.Console.IsOutputRedirected)
                System.Console.Clear();

            CultureInfo.CurrentCulture = CultureInfo.CreateSpecificCulture("en-US");

            System.Console.WriteLine("Sentiment Analysis");
            System.Console.WriteLine("======================\n");

            // load data
            System.Console.WriteLine("Loading data....");
            string fileContent = ReadDataFile(".\\data\\wikipedia-detox-250-line-data.tsv");

            // preprocess file
            System.Console.WriteLine("Processing data....");
            string[,] processedComments = ProcessComments(fileContent);
            System.Console.WriteLine($"Data file contains {processedComments.GetLength(0)} comments\n");
            // for(int i = 0; i < 3; i++)
            // {
            //     System.Console.WriteLine($"{processedComments[i, 0]}\t{processedComments[i, 1]}");
            // }
            // System.Console.WriteLine("...\n");

            // generate the vocabulary list
            System.Console.WriteLine("Generating Vocabulary List....");
            string[] vocab = GenerateVocabulary(processedComments);
            System.Console.WriteLine($"Vocabulary generated with {vocab.Length} words\n");

            // get labels from preprocessed comments
            System.Console.WriteLine("Retrieving labels...");
            Vector<double> Y = GetLables(processedComments);
            //System.Console.WriteLine(Y);

            // extract features from processed comments and vocabulary
            System.Console.WriteLine("Extracting features...");
            Matrix<double> X = GetFeatures(processedComments, vocab);
            //System.Console.WriteLine(X);

            // split the data into train and test in ratio 80:20
            System.Console.WriteLine("Splitting data...");
            int m = X.RowCount;
            int n = X.ColumnCount;
            int testsetSize = m * 20 /100;

            Vector<double> testLabel = Y.SubVector(0, testsetSize);
            Matrix<double> testFeatures = X.SubMatrix(0, testsetSize, 0, n);

            Vector<double> trainingLabel = Y.SubVector(testsetSize, m - testsetSize);
            Matrix<double> trainingFeatures = X.SubMatrix(testsetSize, m - testsetSize, 0, n);

            System.Console.WriteLine();
            System.Console.WriteLine($"Test set: {testLabel.Count}");
            System.Console.WriteLine($"Training set: {trainingLabel.Count}");

            // trainiong SVM
            System.Console.WriteLine("\nTraining linear SVM ...\n");

            // SVM parameters
            double C = .4;
            var linearKernel = KernelHelper.LinearKernel();

            List<List<double>> libSvmData = ConvertToLibSvmFormat(trainingFeatures, trainingLabel);
            svm_problem prob = ProblemHelper.ReadProblem(libSvmData);                        
            var svc = new C_SVC(prob, linearKernel, C);

            System.Console.WriteLine();

            // accuacy on training set
            Vector<double> prediction = SvmPredic(trainingFeatures, svc);
            double accuracy = CalculateAccuracy(prediction, trainingLabel);
            System.Console.WriteLine("Training set Accuracy: {0:f2}%\n", accuracy);


            // accuacy on test set
            prediction = SvmPredic(testFeatures, svc);
            accuracy = CalculateAccuracy(prediction, testLabel);
            System.Console.WriteLine("Test set Accuracy: {0:f2}%\n", accuracy);

            // F1 score
            double f1Score = CalculateF1Score(prediction, testLabel);
            System.Console.WriteLine("F1 Score on test set: {0:f2}%\n", f1Score * 100);

            //Pause();
        }

        private static double CalculateF1Score(Vector<double> prediction, Vector<double> label)
        {
            double precision;
            double recall;
            double f1Score;

            double truePositives = 0;
            double trueNegatives = 0;
            double falseNegatives = 0;
            double falsePositives = 0;

            int m = label.Count;

            for(int i = 0; i < m; i++)
            {
                double predicted = prediction[i];
                double actual = label[i];

                // true positives: predicted 1, actual 1
                if(predicted == 1 && actual == 1)
                    truePositives++;
                
                // true negatives: predicted 0, actual 0
                if(predicted == 0 && actual == 0)
                    trueNegatives++;

                // false negatives: predicted 0, actual 1
                if(predicted == 0 && actual == 1)
                    falseNegatives++;

                // false positives: predicted 1, actual 0
                if(predicted == 1 && actual == 0)
                    falsePositives++;

            }

            precision = truePositives / (truePositives + falsePositives);
            recall = truePositives / (trueNegatives + falseNegatives);

            f1Score = 2 * (precision * recall) / (precision + recall);

            return f1Score;        
        }

        private static double CalculateAccuracy(Vector<double> prediction, Vector<double> label)
        {
            int m = label.Count;

            Vector<double> comp = Vector<double>.Build.Dense(m);

            for(int i = 0; i < m; i++)
            {
                if(prediction[i] != label[i])
                    comp[i] = 0;
                else
                    comp[i] = 1;                
            }

            double accuracy = comp.Mean() * 100;
            return accuracy;
        }

        private static Vector<double> SvmPredic(Matrix<double> X, C_SVC svc)
        {
            int m = X.RowCount;
            int n = X.ColumnCount;
            Vector<double> prediction = Vector<double>.Build.Dense(m);
            for(int i = 0; i < m; i++)
            {
                svm_node[] nodes = new svm_node[n];

                for(int k = 0; k < n; k++)
                {
                    nodes[k] = new svm_node() { index = k + 1, value = X[i, k]};
                }

                prediction[i] = svc.Predict(nodes);
            }

            return prediction;
        }

        private static List<List<double>> ConvertToLibSvmFormat(Matrix<double> x, Vector<double> y)
        {
            List<List<double>> data = new List<List<double>>();

            int m = x.RowCount;

            for(int i = 0; i < m; i++)
            {
                List<double> r = new List<double>();

                r.Add(y[i]);
                foreach(var c in x.Row(i))
                {
                    r.Add(c);
                }
                data.Add(r);
            }

            return data;
        }

        private static Matrix<double> GetFeatures(string[,] processedComments, string[] vocab)
        {
            int m = processedComments.GetLength(0);     // number of examples
            int n = vocab.Length;                       // number of features

            Matrix<double> X = Matrix<double>.Build.Dense(m, n);

            for(int k = 0; k < m; k++)
            {
                string comment = processedComments[k, 1];
 
                for(int i = 0; i < vocab.Length; i++)
                {
                    if(comment.Contains(vocab[i]))
                        X[k, i] = 1;
                    else
                        X[k, i] = 0;
                }                
            }

            return X;

        }

        private static Vector<double> GetLables(string[,] processedComments)
        {
            double[] d = new double[processedComments.GetLength(0)];

            for(int i = 0; i < d.Length; i++)
            {
                d[i] = double.Parse(processedComments[i, 0]);
            }

            Vector<double> Y = Vector<double>.Build.DenseOfArray(d);
            return Y;
        }

        private static string[,] ProcessComments(string fileContent)
        {

            string[] lines = fileContent.Split('\n');
            string[,] processedComments = new string[lines.Length, 2];

            // initialize stemmer
            IStemmer stemmer = new EnglishStemmer();

            for(int idx = 0; idx < lines.Length; idx++)
            {
                // get the line
                string line = lines[idx];

                // sentiment: 0=non-toxic; 1=toxic
                string sentiment = line.Split('\t')[0];
                string comment = line.Split('\t')[1];

                // split the line in words
                comment = comment.ToLower();
                string[] words = comment.Split(' ', '=', '-', ':', '?', '!', '.', ',', '*', '\\', '/', '\"', '\r');

                // remove empty string
                words = words.Where(w => w.Length > 0).ToArray();

                // stemming words
                words = stemmer.GetSteamWords(words);               

                comment = "";
                for(int i = 0; i < words.Length; i++)
                {
                    comment = comment + words[i];
                    if(i < words.Length - 1)
                        comment += " ";
                }
                

                processedComments[idx, 0] = sentiment;
                processedComments[idx, 1] = comment;
            }

            return processedComments;
        }

        private static string[] GenerateVocabulary(string[,] processedComments)
        {
            int wordRecurrency = 5;     // word recurrency
            int minWordLength = 3;      // minimum word length
            int m = processedComments.GetLength(0);

            List<string> wordList = new List<string>();

            for(int i = 0; i < m; i++)
            {
                wordList.AddRange(processedComments[i, 1].Split(' '));
            }

            var words = wordList                    
                            .Where(w => w.Length >= minWordLength)
                            .GroupBy(w => w)
                            .Where(w => w.Count() > wordRecurrency)
                            .Select(w => w.Key)
                            .ToArray();

            // sort
            Array.Sort(words);

            return words;
        }

        private static string ReadDataFile(string path)
        {
            string fileContent;
            using(System.IO.StreamReader sr = new System.IO.StreamReader(path))
            {
                fileContent = sr.ReadToEnd();
            }        
            return fileContent;
        }

        private static void Pause()
        {
            if(!System.Console.IsOutputRedirected)
            {
                Console.WriteLine("\nProgram paused. Press enter to continue.\n");
                Console.ReadKey();
            }
        }        
    }
    
}
