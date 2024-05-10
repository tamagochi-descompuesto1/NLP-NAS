import time
from jtop_stats import jtop_stats
from transformers import ElectraTokenizer, ElectraForCausalLM

def test_model(model_path: str, model, input_text: str) -> str:
    tokenizer = ElectraTokenizer.from_pretrained(model_path)

    input = tokenizer.encode(input_text, return_tensors='pt')
    output = model.generate(input, max_length=100, num_return_sequences=1)

    text = tokenizer.decode(output[0], skip_special_tokens=True)

    return text


if __name__ == '__main__':
    stats = jtop_stats.JtopStats()

    # Load model
    print('Loading model smallbenchnlp/bert-small...')
    model = ElectraForCausalLM.from_pretrained('models/smallbenchnlp/ELECTRA-DeBERTa-Small', ignore_mismatched_sizes=True)
    print('Model loaded.')
            
    # Test model
    print('Model testing enabled. Write yout input text.')
    input_text = input('Your input text: ')

    stats.set_stats()
    output = test_model('smallbenchnlp/ELECTRA-DeBERTa-Small', model, input_text)

    print(output)

    path = 'stat_dumps/ELECTRA-DeBERTa-Small/ELECTRA-DeBERTa-Small-test-' + time.strftime("%Y-%m-%d_%H-%M-%S", time.localtime()) + '.txt' 
    stats.dump_deltas(path)
    print('Hardware stats dumped at', path)