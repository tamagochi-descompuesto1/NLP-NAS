import time
from jtop_stats import jtop_stats
from transformers import RobertaTokenizer, RobertaForCausalLM

def test_model(model_path: str, model, input_text: str) -> str:
    tokenizer = RobertaTokenizer.from_pretrained(model_path)

    input = tokenizer.encode(input_text, return_tensors='pt')
    output = model.generate(input, max_length=100, num_return_sequences=1)

    text = tokenizer.decode(output[0], skip_special_tokens=True)

    return text


if __name__ == '__main__':
    stats = jtop_stats.JtopStats()

    # Load model
    print('Loading model smallbenchnlp/bert-small...')
    model = RobertaForCausalLM.from_pretrained('models/smallbenchnlp/roberta-small')
    print('Model loaded.')
            
    # Test model
    print('Model testing enabled. Write yout input text.')
    input_text = input('Your input text: ')

    stats.set_stats()
    output = test_model('smallbenchnlp/roberta-small', model, input_text)

    print(output)

    path = 'stat_dumps/roberta-small/roberta-small-test-' + time.strftime("%Y-%m-%d_%H-%M-%S", time.localtime()) + '.txt' 
    stats.dump_deltas(path)
    print('Hardware stats dumped at', path)