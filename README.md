# Chatbot_MVP_for_Healthcare_Data

### The deatils of each step in creating a Chatbot using generative AI is provided below.

# Chatbot with Generative Pretrained Transformer(GPT) Model
#### The objective of this project is to develop a Chatbot using a pretrained text generation model (GPT-model) and go over the process of fine-tuning it on the given dataset. This fine-tuning step is key in producing highquality models. Fine-tuning allows us to adapt a model to a specific dataset or domain.

#### The supervised fine-tuning method which is a most common method for fine-tuning text generation models has been implimented. The transformative potential of finetuning pretrained text generation models is to make them more effective tools for the application.

# Install the Required Libraries
#### The required libraries are installed

# Supervised Fine-Tuning (SFT)
#### With supervised fine-tuning (SFT), we can adapt the base model to follow instructions. During this fine-tuning process, the parameters of the base model are updated to be more in line with our target task, like following instructions. Like a pretrained model, it is trained using next-token prediction but instead of only predicting the next token, it does so based on a user input.

# Data Importing and Preprocessing
#### Import the data from CSV file and prepare the data for traing the LLM.

## Models - Quantization
#### We have our data, now we can start loading in our model. This is where we apply the Q in QLoRA, namely quantization. Here I have used the "bitsandbytes" package to compress the pretrained model to a 4-bit representation. In BitsAndBytesConfig, you can define the quantization scheme. I followed the steps used in the original QLoRA paper and load the model in 4-bit (load_in_4bit) with a normalized float representation (bnb_4bit_quant_type) and double quantization (bnb_4bit_use_double_quant).

# Configuration
## LoRA Configuration
#### Next, I have defined defined LoRA configuration using the "peft" library, which represents hyperparameters of the fine-tuning process.

### Parameters
#### r: This is the rank of the compressed matrices. Increasing this value will also increase the sizes of compressed matrices leading to less compression and thereby improved representative power. Values typically range between 4 and 64.

#### lora_alpha: Controls the amount of change that is added to the original weights. In essence, it balances the knowledge of the original model with that of the new task. A rule of thumb is to choose a value twice the size of r.

#### target_modules: Controls which layers to target. The LoRA procedure can choose to ignore specific layers, like specific projection layers. This can speed up training but reduce performance and vice versa.

#### We can experiment by Playing with the parameter values to get an intuitive understanding of values that work and those that do not for our use case.

# Training Configuration
#### Now, we need to configure the training parameters.

### There are several parameters worth mentioning:
#### num_train_epochs: The total number of training rounds. Higher values tend to degrade performance so we generally like to keep this low.

#### learning_rate: Determines the step size at each iteration of weight updates. It is know that higher learning rates work better for larger models (>33B parameters).

#### lr_scheduler_type: A cosine-based scheduler to adjust the learning rate dynamically. It will linearly increase the learning rate, starting from zero, until it reaches the set value. After that, the learning rate is decayed following the values of a cosine function.

#### optim: The paged optimizers used in the original QLoRA paper.

# Model Training

#### Now, we have prepared models and parameters, and  we can start fine-tuning our model. We load in "SFTTrainer" and simply run "trainer.train()". During training the loss will be printed every 10 steps according to the logging_steps parameter. Note: I have used my "Weights&Biases" API key to train the model

# Merge Adapter

#### After trained QLoRA weights, then combine them with the original weights to use them. We reload the model in 16 bits, instead of the quantized 4 bits, to merge the weights. Although the tokenizer was not updated during training, we save it to the same folder as the model for easier access.

# Inference
#### After merging the adapter with the base model, we can use it with the prompt template that is defined earlier.

# Evaluating Generative Models

#### Evaluating generative models poses a significant challenge. Generative models are used across many diverse use cases, making it a challenge to rely on a singular metric for judgment. Given their probabilistic nature, generative models do not necessarily generate consistent outputs. No one metric is perfect for all use cases.

#### One common metrics category for comparing generative models is wordlevel evaluation. These classic techniques compare a reference dataset with the generated tokens on a token(set) level. Common word-level metrics include perplexity, ROUGE, BLEU, and BERTScore.

#### A common method for evaluating generative models on language generation and understanding tasks is on well-known and public benchmarks, such as MMLU,GLUE, TruthfulQA, GSM8k, and HellaSwag. These benchmarks give us information about basic language understanding but also complex analytical answering.

#### Aside from natural language tasks, some models specialize in other domains, like programming. These models tend to be evaluated on different benchmarks, such as HumanEval, which consists of challenging programming tasks for the model to solve.

# Preference-Tuning / Alignment / Reinforcement Learning from Human Feedback (RLHF)

#### Although the present model can now follow instructions, we can further improve its performance by implementing additional fine-tuning technics during the training phase.

#### A common method to fine-tune the LLM with the trained reward model is Proximal Policy Optimization (PPO). PPO is a popular reinforcement technique that optimizes the instruction-tuned LLM by making sure that the LLM does not deviate too much from the expected rewards.

#### A disadvantage of PPO is that it is a complex method that needs to train at least two models, the reward model and the LLM, which can be more costly than perhaps necessary.

#### Direct Preference Optimization (DPO) is an alternative to PPO and does away with the reinforcement-based learning procedure. Instead of using the reward model to judge the quality of a generation, we let the LLM itself do that. Compared to PPO, the DPO method is found to be more stable during training and more accurate.

#### The combination of SFT+DPO is a great way to first fine-tune the model to perform basic chatting and then align its answers with human preference. However, it is computationally expensive since we need to perform two training loops and potentially tweak the parameters in two processes.

#### Recently, a new method, called Odds Ratio Preference Optimization (ORPO) of aligning preferences has developed by J Hong (2024). This method combines SFT and DPO into a single training process. It removes the need to perform two separate training loops, further simplifying the training process while allowing for the use of QLoRA.

# Conclusion

#### In project, I have developed a Chatbot (generative AI model) by fine-tuning pretrained LLM. I have used a lite weight pretrained LLM model ""Qwen/Qwen2-0.5B-Instruct" and fine-tuned on the given data set "mle_screening_dataset.csv". The fine-tuning was performed by making use of parameter-efficient fine-tuning (PEFT) through the low-rank adaptation (LoRA) technique and the LoRA was extended through quantization, a technique for reducing memory constraints when representing the parameters of the model and adapters. The model performance was tested with four different examples. Since I have used a lite weight pretarined model, there is a limitation on the number of characters in text generation.

#### We can further improve the model by utilizing large medical-specialized LLMs such as BioGPT, MedAlpaca, or PMC-LLaMA. For deployment in healthcare (HIPAA-compliant), even we can use better models like Azure OpenAI GPT-4 or Google Med-PaLM 2. Additionally, we can run a RAG pipeline with trusted medical sources to get most accurate answers for the questions asked.
