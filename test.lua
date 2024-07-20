
--@alias MyType = meta<{ x: number }, { tag: string }>

--@type MyType
local myType = setmetatable({ x = 5 }, { tag = false })