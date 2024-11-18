import React from "react";
import Sidebar from "./component/Sidebar";
import MainContent from "./component/MainContent";
import SearchBox from "./component/SearchBox";

function App() {
  return (
    <div className="App flex h-screen bg-gray-100">
      <Sidebar />
      <div className="flex flex-col flex-grow">
        <SearchBox />
        <main className="p-6 flex-grow overflow-y-auto">
          <MainContent />
        </main>
      </div>
    </div>
  );
}

export default App;
