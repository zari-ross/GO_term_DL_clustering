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
