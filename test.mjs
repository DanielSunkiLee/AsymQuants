import { gguf } from "@huggingface/gguf";


const URL_LLAMA = "https://huggingface.co/TheBloke/Llama-2-7B-Chat-GGUF/resolve/191239b/llama-2-7b-chat.Q2_K.gguf";

console.log("1.connect network and start parsing");
const {metadata, tensorInfos} = await gguf(URL_LLAMA);
console.log("2.complete data load");

console.log("3.top3 sample");
console.log(tensorInfos.slice(0,3));
