# 專案介紹 Project Description
此專案的目的，是為了實現本機端中文化語音轉文字之執行，因此採用**OpenAI**的 **Whisper** 模型，加上 **飛槳 PaddlePaddle** 的中文標點符號標註，以完整呈現中文的閱讀性。

# 功能及流程
此專案重點在於接受不同模型的套用限制，因此流程並不是簡單的將兩個模組接起來即可，說明如下：

1. 將影音檔轉為 wav 檔：
   > 轉錄長時間的影音檔(mp4)時，Whisper有高度可能遺失最後30%的內容，因此需要進行將影音專為WAV檔的過程。詳細原因不知，亦有可能是因為本身顯卡效能不佳導致。
    ```python
       def convert_wav(b4cover_file_name, b4cover_file_path,  b4cover_type ):
          b4cover_file = b4cover_file_path + b4cover_file_name + "." + b4cover_type
          
          conver_output_file = b4cover_file_path + b4cover_file_name + ".wav"
          audio = AudioSegment.from_file(b4cover_file)
          
          wav_audio = audio.set_channels(1).set_frame_rate(44100)  
          
          wav_audio.export(conver_output_file, format="wav")
          
          print("轉為wav檔成功：\n" + conver_output_file + "\n")
          return conver_output_file

3. 將wav檔導入Whisper 模型
   > Whisper 模型的model type 可選擇 tiny, base, small, medium, large，但我永遠選擇large，因為精準度是我最重要的考量
    ```python
       def my_whisper(audio,model_type):
          
          print ("開始進行中文語音辦識，請稍等")
          model = whisper.load_model(model_type)
          result = model.transcribe(audio, language='zh')
          print ("中文語音辦識完成")
          return result

5. 將輸出之文檔，轉為簡體中文
   > 此時轉為簡中，主要目的是因為Paddle NLP處理簡中效果較佳
   ```python
      #繁簡轉換涵數( t2s or s2t)
      def Ch_Convert(transcript, method):
          model = OpenCC(method)
          converted = model.convert(transcript)
          return converted
   
6. 進行文字長度處理，導入 Paddle 模型加註標點符號，轉回繁體中文
   > 此時因 Paddle 模型的token有所限制，因此需將文檔的長度切分，並將最後結果轉回繁中
   ```python
      def Add_punc(raw_script):
      
        def split_text(text, max_length):
            return [text[i:i+max_length] for i in range(0, len(text), max_length)]
            
        
        # 整理逐字稿
        def process_text(text):
            model = hub.Module(name='auto_punc')
            Punc_result = model.add_puncs(text)
            return Punc_result
        
        # 處理長字串
        def process_long_text(long_text):
            text_list = split_text(long_text, 300)
            processed_text_list = process_text(text_list) 
            return "".join(processed_text_list)
        
        # 呼叫分段處理函式
        processed_transcript = process_long_text(raw_script)
        print('標點符號加入完成\n')
        return processed_transcript
7. 若有專業文字需調整，利用CSV定義檔，進行文字置換
   > 此步驟可以跳過，如果不需要進行文字的置換
   ```python
     def fix_wording(fix_txt, csv_file):
        replace_dict = {}
        # 將CSV檔轉換為字典
        with open(csv_file, mode='r') as file:
            reader = csv.reader(file)
    
            for row in reader:
                key = row[0]
                value = row[1]
                replace_dict[key] = value
        
        Replaced_text = fix_txt
    
        for old_word, new_word in replace_dict.items():
            Replaced_text = Replaced_text.replace(old_word, new_word)
    
        myfix_result = Replaced_text.replace("-", "\n")
        
        print('文字更正置換完成\n')
        return myfix_result
9. 儲存為純文字txt檔及word檔
   > 最終輸出之檔案
   ```python
      def convert_text(input_text, file_name , output_file_path):
          output_text = output_file_path + file_name + ".txt" 
      
          with open(output_text, 'w', encoding='utf-8') as file:
              file.write(input_text)
      
          print(f"文本已输出到 {output_text} 純文字檔中")
          
      def convert_word(input_text, file_name , output_file_path):
        
          doc = Document()
        
          word_text = input_text
          doc.add_paragraph(word_text)
      
          text_type = "docx"
          output_word = output_file_path + file_name + "." + text_type
          doc.save(output_word)
      
          print(f"文本已输出到 {output_word} 文件中。")
 



# 環境需求 Environment Request
此專案對於環境設置的需求極高，因為Pytorch 及 Paddle為不同框架，在同時運行在同一張顯卡上有GPU資源衝突產生，因此建議參考以下目前2023.11之版本清單

## 硬體
* CPU: Core(TM) i5-13400
* RAM: DDR4 32 GB
* GPU: GeForce RTX™ 4070 WINDFORCE OC 12G

## 套件清單及版本
* Python 3.8
* CUDA 11.8
* CUDNN 8.3
* PyTorch 2.0.1 + cuda118
* Paddlepaddle-gpu 2.4.2 + **cuda117** 
* Paddle NLP **2.5.2**


## 參考連結
* Whisper : https://github.com/openai/whisper
* PyTorch : https://pytorch.org/
* Paddle : https://www.paddlepaddle.org.cn/
* OpenCC (簡體繁體轉換): https://github.com/BYVoid/OpenCC 

