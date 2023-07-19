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
		
rule compute_embeddings:
    input:
        terms="rat_cleaned_terms.json",
        model="mask_go_terms.model",
        pkl="word_embeddings_mask_trained_on_abstracts.pkl"
    output:
        json="rat_cleaned_terms_with_embeddings.json",
        png="mask_model_architecture.png"
    shell:
        """
        python 3_Project_GO_term_embeddings.py {input.terms} {input.model} {input.pkl} {output.json} {output.png}
        """
		
rule compute_tsne:
    input:
        json="rat_cleaned_terms_with_embeddings.json"
    output:
        json="rat_cleaned_terms_with_embeddings_and_tsne.json"
    shell:
        """
        python 4_Project_GO_term_tsne_embeddings.py {input.json} {output.json}
        """
		
rule cluster_terms:
    input: 
        "rat_cleaned_terms_with_embeddings_and_tsne.json"
    output: 
        "rat_cleaned_terms_with_embeddings_clusters_and_tsne.json"
    params:
        inputFile = "rat_cleaned_terms_with_embeddings_and_tsne.json",
        outputFile = "rat_cleaned_terms_with_embeddings_clusters_and_tsne.json"
    shell: 
        "python 5_Project_GO_term_cluster.py --input {params.inputFile} --output {params.outputFile}"
		
rule visualize_clusters:
    input: 
        "rat_cleaned_terms_with_embeddings_clusters_and_tsne.json",
        "GO_terms_Example1.txt"
    output: 
        enrichment_scores = "enrichment_scores.csv",
        cluster_enrichment_plot = "cluster_enrichment.png",
        cluster_visualization_plot = "cluster_visualization.png"
    params:
        num_clusters = 40
    shell: 
        "python 6_Visualize_clusters.py {input} {params.num_clusters} {output.enrichment_scores} {output.cluster_enrichment_plot} {output.cluster_visualization_plot}"

