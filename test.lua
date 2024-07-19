
-- @record Person
-- @field name: string
-- @field age: number

-- @params p: Person
function getName(p)
    return p.name
end

local alice = { name = "Alice", age = 30 }
return getName(alice)
