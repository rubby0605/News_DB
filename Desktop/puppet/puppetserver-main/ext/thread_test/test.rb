require 'net/http'
require 'json'
require 'rspec/expectations'

include RSpec::Matchers

class CatalogTester
  def initialize
    @uri = URI("http://localhost:8140/puppet/v3/catalog/#{@catalog}?environment=#{@environment}")
  end

  def run
    encoded_result = Net::HTTP.get(@uri)
    result = JSON.parse(encoded_result)
    verify(result)
  end

  def verify(_result)
    raise NotImplementedError
  end
end

class CatalogOneTester < CatalogTester
  def initialize
    @catalog = 'testone'
    @environment = 'production'
    super
  end

  def verify(result)
    expect(result).to include('name' => @catalog)
    expect(result).to include('environment' => @environment)
    # v4 function
    expect(result['resources']).to include(a_hash_including('title' => 'do it'))
    # v3 function
    expect(result['resources']).to include(a_hash_including('title' => 'the old way of functions'))
    # class param from hiera
    expect(result['resources']).to include(a_hash_including('parameters' => { 'input' => 'froyo' }))
  end
end

class CatalogTwoTester < CatalogTester
  def initialize
    @catalog = 'testtwo'
    @environment = 'funky'
    super
  end

  def verify(result)
    expect(result).to include('name' => @catalog)
    expect(result).to include('environment' => @environment)
    # color is turned on in color function
    expect(result['resources']).to include(a_hash_including('title' => 'The funky color is ansi'))
    # v4 function
    expect(result['resources']).to include(a_hash_including('title' => 'Always on the one'))
    # v3 function
    expect(result['resources']).to include(a_hash_including('title' => 'old school v3 function'))
    # class param from hiera
    expect(result['resources']).to include(a_hash_including('parameters' => { 'input' => 'hiera_funky' }))
  end
end

class CatalogThreeTester < CatalogTester
  def initialize
    @catalog = 'testthree'
    @environment = 'production'
    super
  end

  def verify(result)
    expect(result).to include('name' => @catalog)
    expect(result).to include('environment' => @environment)
    # function only loaded for this node
    expect(result['resources']).to include(a_hash_including('title' => "I'm a different function"))
  end
end

# Use tester2 for funky env, and swap out for tester1 to use production instead,
# or tester3 for a node in the production env with special classification.
tester2 = CatalogTwoTester.new
Process.fork do
  tester2.run
end

exit_codes = Process.waitall

if exit_codes.all? { |s| s[1].exitstatus.zero? }
  exit 0
else
  exit 1
end
