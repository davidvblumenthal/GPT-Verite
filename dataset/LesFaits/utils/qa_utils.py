import random


def construct_extractive_qa_sample(sample):
    
    question = sample["question"]
    context = sample["context"]#" ".join(sample["context"])
    answer = sample["answers"][0] #mrqa

    """
    def opt_1(question, context, answer):
        sample = "Question: {}; \n\nContext: {};\n\nAnswer: {}".format(question, context, answer)
        return sample
    """
    def opt_2(question, context , answer):
        sample = "Given this context: {};\n\nAnswer this question: {};\n\nAnswer: {}".format(context, question, answer)
        return sample

    """
    def opt_3(question, context, answer):
        sample = f"Based on the following text: {{{context}}}; {{{question}}};\n\nAnswer: {answer}"
        return sample
    """
    prompt_choices = [opt_2]#, opt_1] # opt_3]

    sample = random.choice(prompt_choices)(question, context, answer)

    return {"text": sample}



def contruct_closed_book_qa(sample):

    question = sample["question"]
    answer = sample["answer"]
    
    prompt_choices = [
            "Question: {}\n\nAnswer: {}"
            #"{} \n\nAnswer: {}", 
            #"Answer the following question: {} \nAnswer: {}"
            ]

    sample = random.choice(prompt_choices).format(question, answer)

    return {"text": sample}
    

    
    

