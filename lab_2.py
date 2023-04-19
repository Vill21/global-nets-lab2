### Original hamming code is taken from open git gist page: https://gist.github.com/baskiton/6d361f4155f41e91c4be1dce897f7431

import warnings
warnings.filterwarnings("ignore", category=DeprecationWarning) 

from crc16 import crc16xmodem
from typing import List
from math import log2, ceil
from random import randrange, uniform
import hamming_codec


def __hamming_common(src: List[List[int]], s_num: int, encode=True) -> int:
    s_range = range(s_num)
    errors_corrected = 0

    for i in src:
        sindrome = 0
        for s in s_range:
            sind = 0
            for p in range(2 ** s, len(i) + 1, 2 ** (s + 1)):
                for j in range(2 ** s):
                    if (p + j) > len(i):
                        break
                    sind ^= i[p + j - 1]

            if encode:
                i[2 ** s - 1] = sind
            else:
                sindrome += (2 ** s * sind)

        if (not encode) and sindrome:
            i[sindrome - 1] = int(not i[sindrome - 1])
            errors_corrected += 1

    return errors_corrected

### Deprecated ###
def comparison(msg: str, dec_msg: str, mode: int):
    result_word = 0
    result_letter = 0

    msg_b = msg.encode("utf-8")
    bit_msg_seq = []
    for byte in msg_b:
        bit_msg_seq += list(map(int, f"{byte:08b}"))

    dec_msg_b = dec_msg.encode("utf-8")
    bit_dec_msg_seq = []
    for byte in dec_msg_b:
        bit_dec_msg_seq += list(map(int, f"{byte:08b}"))

    res_len = len(bit_msg_seq) // mode
    res_len = ceil((len(msg_b) * 8) / mode)
    bit_msg_seq += [0] * (res_len * mode - len(bit_msg_seq))
    res_dec_len = len(bit_dec_msg_seq) // mode
    res_dec_len = ceil((len(dec_msg_b) * 8) / mode)
    bit_dec_msg_seq += [0] * (res_dec_len * mode - len(bit_dec_msg_seq))

    equal_sizes = res_len == res_dec_len
    print('Bit representations are of equal sizes:', equal_sizes, res_len, res_dec_len)
    if equal_sizes:
        print('Comparison is valid and bitwise')
    else:
        print('Comparison is invalid because bits of the decoded message are shifted at some point')

    for i in range(min(res_len, res_dec_len)):
        code_msg = bit_msg_seq[i * mode:i * mode + mode]
        code_dec = bit_dec_msg_seq[i * mode:i * mode + mode]
        if code_msg != code_dec:
            result_word += 1
        for i, j in zip(code_dec, code_msg):
            if i != j:
                result_letter += 1

    return result_word, result_letter
###


def comparable(enc_bits: str, dec_bits: str) -> bool:
    return len(enc_bits) == len(dec_bits)


class NonComparableBits(Exception):
    """Raised when bit lengths of encoded and decoded 
    hamming code are not equal
    
    Attributes:
        enc_len: length in bits of the encoded message
        dec_len: length in bits of the decoded message
    """

    def __init__(self, enc_len, dec_len) -> None:
        self.enc_len = enc_len
        self.dec_len = dec_len
        self.message = f"Encoded length {self.enc_len} is not equal to Decoded length {self.dec_len}."
        super().__init__(self.message)


def compare_bits(enc_bits: str, dec_bits: str, mode: int):
    if not comparable(enc_bits, dec_bits):
        raise NonComparableBits(len(enc_bits), len(dec_bits))

    seq_enc = list(map(int, enc_bits))
    seq_dec = list(map(int, dec_bits))
    s_num = ceil(log2(log2(mode + 1) + mode + 1))
    code_len = mode + s_num
    cnt = len(enc_bits) // code_len

    errors_word = 0
    errors_letter = 0

    for i in range(cnt):
        enc_word = seq_enc[i * code_len:i * code_len + code_len]
        dec_word = seq_dec[i * code_len:i * code_len + code_len]

        if enc_word != dec_word:
            errors_word += 1
            for j in range(len(enc_word)):
                if enc_word[j] != dec_word[j]:
                    errors_letter += 1

    return errors_word, errors_letter


def hamming_encode(msg: str, mode: int=8) -> str:
    """
    Encoding the message with Hamming code.
    :param msg: Message string to encode
    :param mode: number of significant bits
    :return: 
    """

    result = ""

    msg_b = msg.encode("utf-8").strip()
    s_num = ceil(log2(log2(mode + 1) + mode + 1))   # number of control bits
    bit_seq = []
    for byte in msg_b:  # get bytes to binary values; every bits store to sublist
        bit_seq += list(map(int, f"{byte:08b}"))

    res_len = ceil((len(msg_b) * 8) / mode)     # length of result (bytes)
    bit_seq += [0] * (res_len * mode - len(bit_seq))    # filling zeros

    to_hamming = []

    for i in range(res_len):    # insert control bits into specified positions
        code = bit_seq[i * mode:i * mode + mode]
        for j in range(s_num):
            code.insert(2 ** j - 1, 0)
        to_hamming.append(code)

    __hamming_common(to_hamming, s_num, True)   # process

    for i in to_hamming:
        result += "".join(map(str, i))

    return result


def hamming_decode(msg: str, mode: int=8):
    """
    Decoding the message with Hamming code.
    :param msg: Message string to decode
    :param mode: number of significant bits
    :return: 
    """

    result = ""

    s_num = ceil(log2(log2(mode + 1) + mode + 1))   # number of control bits
    res_len = len(msg) // (mode + s_num)    # length of result (bytes)
    code_len = mode + s_num     # length of one code sequence

    to_hamming = []

    for i in range(res_len):    # convert binary-like string to int-list
        code = list(map(int, msg[i * code_len:i * code_len + code_len]))
        to_hamming.append(code)

    errors_corrected = __hamming_common(to_hamming, s_num, False)  # process
    
    decoded_bits = ""
    for i in to_hamming:
        decoded_bits += "".join(map(str, i))

    for i in to_hamming:    # delete control bits
        for j in range(s_num):
            i.pop(2 ** j - 1 - j)
        result += "".join(map(str, i))

    msg_l = []

    for i in range(len(result) // 8):   # convert from binary-sring value to integer
        val = "".join(result[i * 8:i * 8 + 8])
        msg_l.append(int(val, 2))

    result = bytes(msg_l).decode("utf-8", errors='ignore')
    
    return result, errors_corrected, decoded_bits


def noizer(msg: str, mode: int, n_errors: int, p: float) -> tuple[str, int]:
    """
    Generates an error in each element of a Hamming encoded message
    """
    seq = list(map(int, msg))
    s_num = ceil(log2(log2(mode + 1) + mode + 1))
    code_len = mode + s_num
    cnt = len(msg) // code_len
    result = ""

    errors_word = 0
    errors_letter = 0
    for i in range(cnt):
        errors_letter_prev = errors_letter
        to_noize = seq[i * code_len:i * code_len + code_len]
        for _ in range(n_errors):
            if uniform(0, 1) < p: # check if an error has occured
                continue
            errors_letter += 1
            noize = randrange(code_len)
            to_noize[noize] = int(not to_noize[noize])
        if errors_letter_prev != errors_letter:
            errors_word += 1
        result += "".join(map(str, to_noize))

    return result, errors_word, errors_letter


if __name__ == "__main__":
    MODE = 63 - 6    # count of signifed bits

    msg = 'Одним из моих любимых развлечений в детстве была родительская печатная машинка. Когда она была свободна, я часами сидел и печатал: книги, брошюры, газеты. Каждое издание выходило тиражом в три экземпляра — под копирку. Иллюстрации и переплёт были выполнены вручную. У меня до сих пор сохранились некоторые забавные образцы моей тогдашней издательской деятельности. Например, однажды я усердно перепечатал довольно объёмную книжку с тестом на IQ. Эта книга была чужая, её нужно было вернуть владельцу через несколько дней. Мне же она настолько понравилась, что захотелось сохранить себе копию. Сканеры и ксероксы, если тогда и существовали, были недоступны. Поэтому пришлось текст перепечатать, а картинки графических заданий аккуратно перерисовать карандашом. Кстати, черчение тоже было моим любимым занятием, поэтому я получил массу удовольствия. Сначала у нас в семье была механическая машинка: приходилось со всей силы лупить по клавишам, чтобы как следует «пробить» три листа, проложенных копиркой. Потом у нас появилась новомодная электрическая модель. Она приятно жужжала внутренним мотором и позволяла печатать безо всяких усилий. Чтобы напечатать букву, достаточно было слегка нажать на клавишу, всё остальное делали внутренние механизмы. Это было уже чем-то похоже на современную компьютерную клавиатуру. Видимо, с тех самых пор у меня сохранилась любовь к простому текстовому формату: безо всяких украшательств, цветов, шрифтов и прочих излишеств. Позже, когда у меня появился полноценный компьютер, мой небольшой винчестер стал постепенно заполняться текстовыми файлами — книгами, инструкциями, моими собственными записями и заметками… Потом постепенно стали появляться файлы других форматов, но до сих пор я стремлюсь по возможности сохранять все тексты в том самом простом и надёжном “plain text”. Я по-прежнему нажимаю клавишу F4 в своём FAR — точно так же, как раньше нажимал её в Norton Commander, а потом в Dos Navigator. Текстовые файлы постепенно обросли форматированием: появились HTML, XML, FB2, MD, DocBook и другие форматы. Но я использую всю эту разметку только по работе или же в исключительных случаях. Простые текстовые файлы безо всякого форматирования много лет остаются на вершине моего личного хит-парада способов хранения информации.'

    def tour(name: str, n: int, p: float, write_file: bool = False):
        if write_file:
            with open('buffer.txt', 'a') as buffer:
                buffer.write(f'{name} tour: n noizes\n\tn = {n}\n\tp = {p}\n\n')

                crc_original = crc16xmodem(msg.encode('utf-8'))
                buffer.write(f'Original checksum {crc_original}\n')

                enc_msg = hamming_encode(msg, MODE)
                enc_msg_crc = crc16xmodem(enc_msg.encode('utf-8'))
                buffer.write(f"Encoded checksum {enc_msg_crc}\n")

                noize_msg, err_word, err_lett = noizer(enc_msg, MODE, n, p)
                buffer.write(f"Errors occured in {err_word} words and in {err_lett} bits overall\n")

                noize_msg_crc = crc16xmodem(noize_msg.encode('utf-8'))
                buffer.write(f"Noized checksum {noize_msg_crc}\n")

                dec_msg, errors, dec_bits = hamming_decode(noize_msg, MODE)
                buffer.write(f'Corrected during hamming correction: {errors}\n')

                dec_msg_crc = crc16xmodem(dec_msg.encode('utf-8'))
                buffer.write(f"Decoded checksum {dec_msg_crc} and Original checksum {crc_original}\n")

                result = 'equal' if dec_msg_crc == crc_original else 'NOT equal'
                buffer.write('Texts are {}\n'.format(result))

                err_word, err_lett = compare_bits(enc_msg, dec_bits, MODE)
                buffer.write(f'Errors between encoded and decoded messages after correction {err_word}, {err_lett}\n\n')
        else:
            print(f'\n\t{name} tour: n noizes\n\tn = {n}\n\tp = {p}\n')

            crc_original = crc16xmodem(msg.encode('utf-8'))
            print(f'Original checksum {crc_original}')
            
            enc_msg = hamming_encode(msg, MODE)
            enc_msg_crc = crc16xmodem(enc_msg.encode('utf-8'))
            print(f"Encoded checksum {enc_msg_crc}")
            
            noize_msg, err_word, err_lett = noizer(enc_msg, MODE, n, p)
            print(f"Errors occured in {err_word} words and in {err_lett} bits overall")
            
            noize_msg_crc = crc16xmodem(noize_msg.encode('utf-8'))
            print(f"Noized checksum {noize_msg_crc} and Original checksum {crc_original}")
            
            dec_msg, errors, dec_bits = hamming_decode(noize_msg, MODE)
            print('Corrected during hamming correction:', errors)
            
            dec_msg_crc = crc16xmodem(dec_msg.encode('utf-8'))
            print(f"Decoded checksum {dec_msg_crc} and Original checksum {crc_original}")
            
            result = 'equal' if dec_msg_crc == crc_original else 'NOT equal'
            print('Texts are {}'.format(result))

            err_word, err_lett = compare_bits(enc_msg, dec_bits, MODE)
            print('Errors between encoded and decoded messages after correction', err_word, err_lett)

    tour('First', 0, 1, True)
    tour('Second', 1, 1, True)
    tour('Third', 1, 0, True)
    tour('Fourth', 1, 0.2, True)
    tour('Fifth', 1, 0.6, True)
    tour('Sixth', 3, 0.5, True)
    tour('Seventh', 7, 0.3, True)

    print('buffer was written successfully')
    print('created buffer.txt with results')
