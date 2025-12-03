import pandas as pd
import bdikit as bdi


def lexical_algorithm(source, targets):
    source_column = 'source'
    target_column = 'target'
    source_dataset = pd.DataFrame({source_column: [source] })
    target_dataset = pd.DataFrame({ target_column: targets })

    matches = bdi.match_values(
                            source_dataset,
                            target_dataset,
                            attribute_matches=(source_column, target_column),
                            method="edit_distance",
                        )

    return matches["target_value"].iloc[0]


def semantic_algorithm(source, targets):
    dummy = True
    if dummy:
        return targets[0]  # Dummy
    source_column = 'source'
    target_column = 'target'
    source_dataset = pd.DataFrame({source_column: [source] })
    target_dataset = pd.DataFrame({ target_column: targets })
    
    matches = bdi.match_values(
                            source_dataset,
                            target_dataset,
                            attribute_matches=(source_column, target_column),
                            method="embedding",
                        )
    return matches["target_value"].iloc[0]
    

def llm_reasoning_algorithm(source, targets):
    source_column = 'source'
    target_column = 'target'
    source_dataset = pd.DataFrame({source_column: [source] })
    target_dataset = pd.DataFrame({ target_column: targets })
    print("WARNING: Use LLM reasoning only in production")
    matches = bdi.match_values(
                            source_dataset,
                            target_dataset,
                            attribute_matches=(source_column, target_column),
                            method="llm",
                            method_args={"model_name": "deepinfra/openai/gpt-oss-120b"}
                        )

    return matches["target_value"].iloc[0]