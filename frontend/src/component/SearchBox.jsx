import axios from "axios";
import { useState } from "react";

function SearchBox() {
  const [Query, setQuery] = useState("");
  const [Response, setResponse] = useState("");
  const [Loading, setLoading] = useState(false);
  const handleSubmit = async (e) => {
    e.preventDefault();
    setLoading(true);
    try {
      const res = await axios.post("http://127.0.0.1:8000/search/", {
        query: Query,
      });
      setResponse(res.data.response);
    } catch (err) {
      console.log(err);
    } finally {
      setLoading(false);
    }
  };
  return (
    <header className="bg-white shadow p-4 flex items-center justify-between">
      <h1 className="text-2xl font-bold text-gray-800">AI 帮你理解科学</h1>
      <form className="flex items-center" onSubmit={handleSubmit}>
        <input
          value={Query}
          type="text"
          onChange={(e) => setQuery(e.target.value)}
          placeholder="请输入要搜索的内容"
          className="border border-gray-300 rounded-lg p-2 w-96 mr-4"
        ></input>
        <button
          type="submit"
          disabled={Loading}
          className="bg-blue-500 text-white py-2 px-4 rounded-lg"
        >
          搜索
        </button>
      </form>
      <div className="response">
        {Response && (
          <div className="mt-6 p-4 border border-gray-300 rounded bg-gray-50">
            <h3 className="text-lg font-semibold mb-2">Result:</h3>
            <p className="text-gray-700">{Response}</p>
          </div>
        )}
      </div>
    </header>
  );
}
export default SearchBox;
