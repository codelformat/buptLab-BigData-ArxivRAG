import React, { useState } from "react";

function MainContent() {
  const [results] = useState([
    {
      image: "https://via.placeholder.com/100", // 替换为实际图片链接
      title: "FinanceBench: A New Benchmark for Financial Question Answering",
      authors: "Pranab Islam, Anand Kannappan",
      published: "Cornell University (2023)",
      citations: 209,
      downloads: 100,
    },
    {
      image: "https://via.placeholder.com/100", // 替换为实际图片链接
      title: "Qwen2.5-Coder Technical Report",
      authors: "Binyuan Hui, Jian Yang, Zeyu Cai",
      published: "arXiv (2023)",
      citations: 0,
      downloads: 100,
    },
    {
      image: "https://via.placeholder.com/100", // 替换为实际图片链接
      title: "Taming Rectified Flow for Inversion and Editing",
      authors: "Jiangshan Wang, Junfu Pu, Zhongqi Qi",
      published: "Cornell University (2024)",
      citations: 150,
      downloads: 250,
    },
  ]);

  return (
    <div className="bg-white p-6 rounded-lg shadow">
      <h2 className="text-xl font-semibold mb-4">以下为热门内容推荐：</h2>
      <div className="grid grid-cols-1 sm:grid-cols-2 lg:grid-cols-3 gap-6">
        {results.map((item, index) => (
          <div
            key={index}
            className="p-4 border border-gray-300 rounded-lg hover:bg-gradient-to-r from-blue-100 to-blue-200 transition-all duration-300"
          >
            {/* 图片部分 */}
            <img
              src={item.image}
              alt={item.title}
              className="w-full h-40 object-cover rounded mb-4"
            />
            {/* 内容部分 */}
            <h3 className="text-lg font-semibold text-blue-600">
              {item.title}
            </h3>
            <p className="text-gray-700">{item.authors}</p>
            <p className="text-gray-500">{item.published}</p>
            <div className="mt-4 text-gray-700">
              <p>引用: {item.citations}</p>
              <p>下载: {item.downloads}</p>
            </div>
            <button className="mt-4 bg-blue-500 text-white py-2 px-4 rounded-lg w-full">
              下载全文
            </button>
          </div>
        ))}
      </div>
    </div>
  );
}

export default MainContent;
