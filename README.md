# Deep Learning for Drug-Target Interaction Prediction

![License](https://img.shields.io/badge/license-MIT-blue.svg)
![Python](https://img.shields.io/badge/python-3.7%2B-brightgreen.svg)
![TensorFlow](https://img.shields.io/badge/tensorflow-2.x-orange.svg)

## აბსტრაქტი
მიზანი: ახალი (Drug-Target, DT) ურთიერთქმედებების იდენტიფიკაცია წამლის აღმოჩენის პროცესის მნიშვნელოვანი ნაწილია. DT ურთიერთქმედებების პროგნოზირებისთვის გამოყენებული მეთოდების უმეტესობა ფოკუსირებულია Binary კლასიფიკაციაზე, სადაც მიზანია განისაზღვროს, ურთიერთქმედებს თუ არა DT წყვილი. თუმცა, ცილა-ლიგანდის ურთიერთქმედებები გულისხმობს შეკავშირების (binding) სიძლიერის მნიშვნელობების უწყვეტობას, რასაც ასევე შეკავშირების აფინურობა ეწოდება და ამ მნიშვნელობის პროგნოზირება კვლავ გამოწვევად რჩება. DT ბაზებში ხელმისაწვდომი აფინურობის მონაცემების მოცულობის ზრდა საშუალებას იძლევა binding affinity - ის პროგნოზირებაში გამოყენებულ იქნას ისეთი მიდგომები, როგორიცაა Deep Learning. ამ კვლევაში ჩვენ გთავაზობთ Deep Learning-ზე დაფუძნებულ მოდელს, რომელიც იყენებს მხოლოდ სამიზნეებისა და წამლების sequential ინფორმაციას DT ურთიერთქმედების შეკავშირების აფინურობის პროგნოზირებისთვის. DT შეკავშირების აფინურობის პროგნოზირებაზე ფოკუსირებული რამდენიმე კვლევა იყენებს ცილა-ლიგანდის კომპლექსების 3D სტრუქტურებს ან ნაერთების 2D მახასიათებლებს. ამ ნაშრომში გამოყენებული ერთ-ერთი ახალი მიდგომაა ცილების თანმიმდევრობებისა და ნაერთების 1D წარმოდგენების მოდელირება რეკურენტული ნეირონული ქსელების (RNN) საშუალებით.

## Introduction

Drug discovery is a complex and costly process, with the identification of drug-target interactions (DTIs) serving as a critical step. As the pharmaceutical landscape evolves, there's growing interest not only in discovering new drugs but also in repurposing existing ones and finding novel interaction partners for approved medications. Accurate prediction of drug-target binding affinities can significantly accelerate this process by narrowing down the vast compound search space.

In recent years, deep learning approaches have shown great promise in predicting drug-target interactions. While many previous studies relied on complex feature engineering or 3D structural information, our work focuses on leveraging the raw sequence data of both proteins and compounds. We propose a series of deep learning models that use only the amino acid sequences of proteins and the SMILES (Simplified Molecular Input Line Entry System) representations of compounds to predict binding affinities.

In this study, we explore and compare several architectures, including simple LSTM, bidirectional LSTM, LSTM with cross-attention, and Transformer with cross-attention. These models aim to capture the complex patterns and long-range dependencies in both protein sequences and chemical structures that contribute to binding affinity. By using only sequence information, our approach offers a more generalizable and scalable solution that doesn't rely on the availability of 3D structural data.

We evaluate our models using the Davis Kinase binding affinity dataset (Davis et al., 2011) and compare their performance in terms of Mean Squared Error (MSE) and Concordance Index (CI). Our results demonstrate the potential of sequence-based deep learning methods in the field of drug discovery and provide insights into the relative strengths of different neural network architectures for this task.

## Materials and Methods

### Dataset 

We used the Davis Kinase dataset (Davis et al., 2011) for training and evaluating our model. This dataset contains binding affinities (Kd values) for interactions between 442 proteins and 68 ligands, totaling 30,056 drug-target interactions. The Kd values were transformed into log space (pKd) using the formula:

pKd = -log10(Kd / 1e9)

### Input Representation 

- For proteins, we used the amino acid sequences obtained from the UniProt database. Each amino acid was encoded using an integer, resulting in a sequence of integers representing the protein.
- For compounds, we used the SMILES (Simplified Molecular Input Line Entry System) representations. Each character in the SMILES string was encoded as an integer, creating a sequence of integers representing the compound.

### Evaluation Metrics 

We evaluated our model using two primary metrics:
1. Concordance Index (CI): This measures the probability that the model correctly ranks the binding affinity of two random drug-target pairs.
2. Mean Squared Error (MSE): This measures the average squared difference between predicted and actual pKd values.

### Model Architecture

We implemented and compared four different model architectures, each designed to capture different aspects of the protein-ligand interaction:

1. SimpleLSTM: This model uses single-layer LSTMs for both drug and protein sequences, followed by a reduced number of fully connected layers. It serves as a baseline to assess the effectiveness of more complex architectures.

2. BidirectionalLSTM: Building on the simple LSTM, this model uses bidirectional LSTMs to capture information from both directions of the sequences. The final hidden states from both directions are concatenated and passed through fully connected layers.

3. LSTMWithCrossAttention: This model introduces a cross-attention mechanism on top of bidirectional LSTMs. After processing the sequences, it applies attention between drug and protein representations, allowing each to focus on relevant parts of the other. The attention outputs are then pooled and passed through fully connected layers.

4. TransformerWithCrossAttention: Our most sophisticated model replaces LSTMs with Transformer encoders. It uses positional encoding and self-attention within each sequence, followed by cross-attention between drug and protein representations. This architecture is designed to capture complex, long-range dependencies in both sequences.

All models use embedding layers to convert integer-encoded sequences into dense vector representations. The final output in each case is a single neuron predicting the binding affinity (pKd value).

These architectures represent a progression from simple sequence processing to more complex attention-based mechanisms, allowing us to assess the impact of different modeling choices on binding affinity prediction performance.

### Comparison with CNN-based Approach 

We compared our RNN-based model with the CNN-based approach described in the DeepDTA paper (Öztürk et al., 2018). Both models were trained and evaluated on the same Davis dataset using identical data splitting and preprocessing steps to ensure a fair comparison.

### Implementation

Our model was implemented using TensorFlow 2.x. We used the Keras API for building and training the neural network. All experiments were conducted on Google Colab's cloud infrastructure, utilizing an NVIDIA A100 GPU. This high-performance computing environment allowed for efficient training and evaluation of our model.

## Results

We evaluated four RNN-based architectures on the Davis Kinase dataset:

| Model                         | CI     | MSE    |
|-------------------------------|--------|--------|
| SimpleLSTM                    | 0.8174 | 0.4833 |
| BidirectionalLSTM             | 0.8060 | 0.5097 |
| LSTMWithCrossAttention        | 0.8725 | 0.2820 |
| TransformerWithCrossAttention | 0.8597 | 0.3285 |
| CNN-based (DeepDTA)           | 0.878  | 0.261  |

The CNN-based model from the original paper achieved: CI - 0.878, MSE - 0.261

Our best performing model, the LSTM with Cross-Attention, approached but did not surpass the CNN-based approach.

## Conclusion

This study explored various RNN-based architectures for predicting drug-target binding affinities using only sequence data. Our results show that RNN-based models, particularly those with attention mechanisms, can perform competitively with CNN-based approaches, though they did not surpass the original results.

The LSTM with Cross-Attention model performed best among our tested architectures, likely due to its ability to capture long-range dependencies and focus on relevant parts of both drug and protein sequences through its bidirectional processing and cross-attention mechanism.

It's important to note that our experiments were limited by GPU access constraints. With more extensive hyperparameter tuning and longer training times, these models have the potential for significant improvement. Future work could explore this potential, as well as investigate hybrid architectures that combine the strengths of both CNNs and RNNs.

In conclusion, while our RNN-based models didn't outperform the CNN approach, they demonstrated the viability of using sequence-based RNN architectures for this task. With further optimization, these models could potentially offer competitive or superior performance, especially in scenarios where only sequence data is available.

## Future Work

1. Extensive hyperparameter tuning to optimize model performance
2. Exploration of hybrid CNN-RNN architectures to combine strengths of both approaches
3. Investigation of performance on larger, more diverse datasets
4. Application of these models to other drug discovery tasks, such as virtual screening

## References

1. Davis, M. I., Hunt, J. P., Herrgard, S., Ciceri, P., Wodicka, L. M., Pallares, G., ... & Zarrinkar, P. P. (2011). Comprehensive analysis of kinase inhibitor selectivity. Nature biotechnology, 29(11), 1046-1051.

2. Öztürk, H., Özgür, A., & Ozkirimli, E. (2018). DeepDTA: deep drug–target binding affinity prediction. Bioinformatics, 34(17), i821-i829.

## Contact

For questions or collaborations, please open an issue in this repository.
