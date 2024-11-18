import React from "react";

function Sidebar() {
  return (
    <aside className="bg-gray-800 text-white w-64 p-6 flex flex-col space-y-6">
      <button className="hover:bg-gray-700 p-2 rounded text-left">
        <span className="text-lg">主页</span>
      </button>
      <button className="hover:bg-gray-700 p-2 rounded text-left">
        <span className="text-lg">推荐</span>
      </button>
      <button className="hover:bg-gray-700 p-2 rounded text-left">
        <span className="text-lg">热点</span>
      </button>
      <button className="hover:bg-gray-700 p-2 rounded text-left">
        <span className="text-lg">期刊/会议</span>
      </button>
    </aside>
  );
}

export default Sidebar;
