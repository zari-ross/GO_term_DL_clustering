rule preprocess_go_terms:
    input:
        obo="go-basic.obo",
        gaf="rgd.gaf"
    output:
        json="rat_cleaned_terms.json"
    shell:
        """
        python 1_Project_GO_term_preprocess_all_part1.py {input.obo} {input.gaf} {output.json}
        """
		
rule preprocess_abstracts:
    input:
        "abstracts.json"
    output:
        "cleaned_abstracts.json"
    shell:
        """
        python 1_Project_GO_term_preprocess_all_part2.py {input} {output}
        """

rule train_lstm_model:
    input:
        "cleaned_abstracts.json"
    output:
        log="GO_term_training.log",
        model="mask_go_terms.model",
        pkl="word_embeddings_mask_trained_on_abstracts.pkl"
    shell:
        """
        python 2_Project_GO_term_train_LSTM_model.py {input} {output.log} {output.model} {output.pkl}
        """