def text_chunking(tokens, chunk_size, overlap):
    """
    Split tokens into fixed-size chunks with optional overlap.
    """
    # Write code here
    output = []
    step = chunk_size - overlap

    for start in range(0, len(tokens), step):
        output.append(tokens[start: start + chunk_size])
        if start + chunk_size == len(tokens):
            break

    return output
        

    
        
        

        
        