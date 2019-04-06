using Microsoft.Data.DataView;
using Microsoft.ML;
using Microsoft.ML.Data;
using System;

// CS0649 compiler warning is disabled because some fields are only 
// assigned to dynamically by ML.NET at runtime
#pragma warning disable CS0649

namespace myApp
{
    class Program
    {
        // STEP 1: Define your data structures
        // IrisData is used to provide training data, and as
        // input for prediction operations
        // - First 4 properties are inputs/features used to predict the label
        // - Label is what you are predicting, and is only set when training
        public class TransactionData
        {
            [LoadColumn(1)]
            public float Amount;

            [LoadColumn(4)]
            public string Label;
        }

        // IrisPrediction is the result returned from prediction operations
        public class TransactionPrediction
        {
            [ColumnName("PredictedLabel")]
            public string PredictedLabels;
        }

        static void Main(string[] args)
        {
            // STEP 2: Create a ML.NET environment  
            MLContext mlContext = new MLContext();

            // If working in Visual Studio, make sure the 'Copy to Output Directory'
            // property of iris-data.txt is set to 'Copy always'
            IDataView trainingDataView = mlContext.Data.LoadFromTextFile<TransactionData>(path: "TransactionHistory.csv", hasHeader: false, separatorChar: ',');

            // STEP 3: Transform your data and add a learner
            // Assign numeric values to text in the "Label" column, because only
            // numbers can be processed during model training.
            // Add a learning algorithm to the pipeline. e.g.(What type of iris is this?)
            // Convert the Label back into original text (after converting to number in step 3)
            var pipeline = mlContext.Transforms.Conversion.MapValueToKey("Label")
                .Append(mlContext.Transforms.Concatenate("Features", "Amount"))
                .AppendCacheCheckpoint(mlContext)
                .Append(mlContext.MulticlassClassification.Trainers.StochasticDualCoordinateAscent(labelColumnName: "Label", featureColumnName: "Features"))
                .Append(mlContext.Transforms.Conversion.MapKeyToValue("PredictedLabel"));

            // STEP 4: Train your model based on the data set  
            var model = pipeline.Fit(trainingDataView);

            // STEP 5: Use your model to make a prediction
            // You can change these numbers to test different predictions
            var prediction = model.CreatePredictionEngine<TransactionData, TransactionPrediction>(mlContext).Predict(
                new TransactionData()
                {
                    Amount = -3.3f,
                });

            Console.WriteLine($"Predicted transaction type is: {prediction.PredictedLabels}");

            Console.WriteLine("Press any key to exit....");
            Console.ReadLine();
        }
    }
}